from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

SamplerMode = Literal["independent", "concat_pair_bins"]


@dataclass(frozen=True)
class PairFrameSamplerCfg:
  num_bins: int = 32
  mode: SamplerMode = "independent"
  adaptive_alpha: float = 1.0e-3
  adaptive_uniform_ratio: float = 0.1
  min_weight: float = 1.0e-4


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
    self.bin_failure_scores = torch.zeros(
      (self.num_pairs, cfg.num_bins),
      dtype=torch.float32,
      device=self.device,
    )
    self.bin_weights = self._bin_probabilities_from_scores()
    self.concat_bin_weights = self.joint_bin_probabilities.reshape(-1)

  @property
  def pair_probabilities(self) -> torch.Tensor:
    return self.pair_weights

  @property
  def joint_bin_probabilities(self) -> torch.Tensor:
    return self.pair_weights[:, None] * self.bin_weights

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
    if not bool(torch.any(failure_mask)):
      return

    failed_pairs = pair_indices[failure_mask]
    failed_bins = self._bin_indices_for_frames(
      failed_pairs,
      local_frames[failure_mask],
    )
    counts = torch.zeros_like(self.bin_failure_scores)
    counts.index_put_(
      (failed_pairs, failed_bins),
      torch.ones_like(failed_bins, dtype=counts.dtype),
      accumulate=True,
    )
    row_sums = torch.sum(counts, dim=1, keepdim=True)
    normalized_counts = torch.zeros_like(counts)
    nonzero_rows = row_sums.squeeze(1) > 0.0
    normalized_counts[nonzero_rows] = counts[nonzero_rows] / row_sums[nonzero_rows]
    self.bin_failure_scores = (
      (1.0 - self.cfg.adaptive_alpha) * self.bin_failure_scores
      + self.cfg.adaptive_alpha * normalized_counts
    )
    self._refresh_probabilities()

  def _bin_probabilities_from_scores(self) -> torch.Tensor:
    floor = self.cfg.adaptive_uniform_ratio / float(self.cfg.num_bins)
    return _normalize_rows(
      self.bin_failure_scores + floor,
      min_weight=self.cfg.min_weight,
    )

  def _refresh_probabilities(self) -> None:
    self.bin_weights = self._bin_probabilities_from_scores()
    self.concat_bin_weights = self.joint_bin_probabilities.reshape(-1)

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
  if float(torch.sum(weights)) <= 0.0:
    weights = torch.ones_like(weights)
  return weights / torch.sum(weights)


def _normalize_rows(weights: torch.Tensor, *, min_weight: float) -> torch.Tensor:
  weights = torch.clamp(weights, min=min_weight)
  row_sums = torch.sum(weights, dim=1, keepdim=True)
  zero_rows = row_sums <= 0.0
  if bool(torch.any(zero_rows)):
    weights = torch.where(zero_rows, torch.ones_like(weights), weights)
    row_sums = torch.sum(weights, dim=1, keepdim=True)
  return weights / row_sums


__all__ = [
  "PairFrameSample",
  "PairFrameSampler",
  "PairFrameSamplerCfg",
  "SamplerMode",
]
