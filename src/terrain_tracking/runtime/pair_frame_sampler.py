from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

SamplerMode = Literal["independent", "concat_pair_bins"]


@dataclass(frozen=True)
class PairFrameSamplerCfg:
  num_bins: int = 32
  mode: SamplerMode = "independent"
  ema_alpha: float = 0.001
  uniform_ratio: float = 0.1
  min_weight: float = 0.0


@dataclass(frozen=True)
class PairFrameSample:
  pair_indices: torch.Tensor
  local_frames: torch.Tensor
  bin_indices: torch.Tensor


class PairFrameSampler:
  def __init__(
    self,
    *,
    frame_counts: torch.Tensor,
    cfg: PairFrameSamplerCfg,
    device: str | torch.device,
    pair_weights: torch.Tensor | None = None,
  ) -> None:
    if cfg.num_bins <= 0:
      raise ValueError("num_bins must be positive")
    if cfg.mode not in ("independent", "concat_pair_bins"):
      raise ValueError(f"unsupported pair sampler mode: {cfg.mode!r}")
    self.cfg = cfg
    self.device = torch.device(device)
    self.frame_counts = frame_counts.to(device=self.device, dtype=torch.long)
    if torch.any(self.frame_counts <= 0):
      raise ValueError("frame_counts must be positive")
    self.num_pairs = int(self.frame_counts.numel())
    if self.num_pairs == 0:
      raise ValueError("at least one pair is required")

    if pair_weights is None:
      pair_weights = torch.ones(self.num_pairs, device=self.device)
    self.pair_weights = _normalize(
      pair_weights.to(device=self.device, dtype=torch.float32),
      min_weight=cfg.min_weight,
    )
    self.bin_weights = torch.full(
      (self.num_pairs, cfg.num_bins),
      1.0 / cfg.num_bins,
      dtype=torch.float32,
      device=self.device,
    )
    self.bin_failed_scores = torch.zeros_like(self.bin_weights)
    self.concat_bin_weights = torch.full(
      (self.num_pairs * cfg.num_bins,),
      1.0 / (self.num_pairs * cfg.num_bins),
      dtype=torch.float32,
      device=self.device,
    )
    self.concat_failed_scores = torch.zeros_like(self.concat_bin_weights)

  def sample(self, env_ids: torch.Tensor) -> PairFrameSample:
    env_ids = env_ids.to(device=self.device)
    num_samples = int(env_ids.numel())
    if self.cfg.mode == "concat_pair_bins":
      flat_bins = torch.multinomial(
        self.concat_bin_weights,
        num_samples,
        replacement=True,
      )
      pair_indices = torch.div(flat_bins, self.cfg.num_bins, rounding_mode="floor")
      bin_indices = flat_bins % self.cfg.num_bins
    else:
      pair_indices = torch.multinomial(
        self.pair_weights,
        num_samples,
        replacement=True,
      )
      bin_indices = torch.multinomial(
        self.bin_weights[pair_indices],
        1,
        replacement=True,
      ).squeeze(-1)
    local_frames = self._sample_local_frames(pair_indices, bin_indices)
    return PairFrameSample(
      pair_indices=pair_indices,
      local_frames=local_frames,
      bin_indices=bin_indices,
    )

  def update_failures(
    self,
    pair_indices: torch.Tensor,
    local_frames: torch.Tensor,
    failure_mask: torch.Tensor,
  ) -> None:
    pair_indices = pair_indices.to(device=self.device, dtype=torch.long)
    local_frames = local_frames.to(device=self.device, dtype=torch.long)
    failure_mask = failure_mask.to(device=self.device, dtype=torch.bool)
    if int(pair_indices.numel()) == 0:
      return

    failed_pairs = pair_indices[failure_mask]
    failed_bins = self._bin_indices_for_frames(
      failed_pairs,
      local_frames[failure_mask],
    )
    counts = torch.zeros_like(self.bin_weights)
    counts.index_put_(
      (failed_pairs, failed_bins),
      torch.ones_like(failed_bins, dtype=counts.dtype),
      accumulate=True,
    )
    if self.cfg.mode == "concat_pair_bins":
      flat_counts = counts.reshape(-1)
      self.concat_failed_scores = _blend_scores(
        self.concat_failed_scores,
        flat_counts,
        alpha=self.cfg.ema_alpha,
      )
      self.concat_bin_weights = _scores_to_probabilities(
        self.concat_failed_scores,
        uniform_ratio=self.cfg.uniform_ratio,
        min_weight=self.cfg.min_weight,
      )
    else:
      for pair_index in torch.unique(pair_indices):
        idx = int(pair_index)
        self.bin_failed_scores[idx] = _blend_scores(
          self.bin_failed_scores[idx],
          counts[idx],
          alpha=self.cfg.ema_alpha,
        )
        self.bin_weights[idx] = _scores_to_probabilities(
          self.bin_failed_scores[idx],
          uniform_ratio=self.cfg.uniform_ratio,
          min_weight=self.cfg.min_weight,
        )

  def sampling_metrics(self) -> tuple[torch.Tensor, torch.Tensor]:
    if self.cfg.mode == "concat_pair_bins":
      entropy = _normalized_entropy(self.concat_bin_weights)
      top1_prob = torch.max(self.concat_bin_weights)
      return entropy, top1_prob

    entropies = torch.stack(
      [_normalized_entropy(weights) for weights in self.bin_weights],
      dim=0,
    )
    return torch.mean(entropies), torch.max(self.bin_weights)

  def _sample_local_frames(
    self,
    pair_indices: torch.Tensor,
    bin_indices: torch.Tensor,
  ) -> torch.Tensor:
    frame_counts = self.frame_counts[pair_indices]
    starts = torch.div(
      bin_indices * frame_counts,
      self.cfg.num_bins,
      rounding_mode="floor",
    )
    ends = torch.div(
      (bin_indices + 1) * frame_counts,
      self.cfg.num_bins,
      rounding_mode="floor",
    )
    ends = torch.maximum(ends, starts + 1)
    widths = ends - starts
    offsets = torch.floor(torch.rand_like(widths, dtype=torch.float32) * widths).long()
    return torch.minimum(starts + offsets, frame_counts - 1)

  def _bin_indices_for_frames(
    self,
    pair_indices: torch.Tensor,
    local_frames: torch.Tensor,
  ) -> torch.Tensor:
    frame_counts = self.frame_counts[pair_indices]
    bins = torch.div(
      local_frames * self.cfg.num_bins,
      frame_counts,
      rounding_mode="floor",
    )
    return torch.clamp(bins, max=self.cfg.num_bins - 1)


def _normalize(weights: torch.Tensor, *, min_weight: float) -> torch.Tensor:
  weights = torch.clamp(weights, min=min_weight)
  total = torch.sum(weights)
  if float(total) <= 0.0:
    return torch.full_like(weights, 1.0 / int(weights.numel()))
  return weights / total


def _blend_scores(
  old_scores: torch.Tensor,
  counts: torch.Tensor,
  *,
  alpha: float,
) -> torch.Tensor:
  return (1.0 - alpha) * old_scores + alpha * counts


def _scores_to_probabilities(
  scores: torch.Tensor,
  *,
  uniform_ratio: float,
  min_weight: float,
) -> torch.Tensor:
  baseline = torch.full_like(scores, uniform_ratio / int(scores.numel()))
  return _normalize(scores + baseline, min_weight=min_weight)


def _normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
  if int(probabilities.numel()) <= 1:
    return torch.ones((), dtype=probabilities.dtype, device=probabilities.device)
  entropy = -(probabilities * (probabilities + 1.0e-12).log()).sum()
  return entropy / torch.log(
    torch.tensor(float(probabilities.numel()), device=probabilities.device)
  )


__all__ = [
  "PairFrameSample",
  "PairFrameSampler",
  "PairFrameSamplerCfg",
  "SamplerMode",
]
