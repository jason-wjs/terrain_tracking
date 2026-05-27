from __future__ import annotations

import torch

from terrain_tracking.runtime.pair_frame_sampler import (
  PairFrameSampler,
  PairFrameSamplerCfg,
)


def test_pair_frame_sampler_independent_mode_samples_valid_frames_and_updates_weights():
  sampler = PairFrameSampler(
    frame_counts=torch.tensor([10, 20, 30]),
    cfg=PairFrameSamplerCfg(num_bins=5, mode="independent"),
    device="cpu",
  )

  sample = sampler.sample(torch.arange(64))

  assert sample.pair_indices.shape == (64,)
  assert torch.all(sample.pair_indices >= 0)
  assert torch.all(sample.pair_indices < 3)
  assert torch.all(sample.local_frames >= 0)
  assert torch.all(sample.local_frames < sampler.frame_counts[sample.pair_indices])

  failure_mask = torch.zeros(64, dtype=torch.bool)
  failure_mask[:8] = True
  sampler.update_failures(sample.pair_indices, sample.local_frames, failure_mask)

  torch.testing.assert_close(
    sampler.bin_weights.sum(dim=1),
    torch.ones(3),
  )


def test_pair_frame_sampler_accumulates_duplicate_failure_bins():
  sampler = PairFrameSampler(
    frame_counts=torch.tensor([10]),
    cfg=PairFrameSamplerCfg(
      num_bins=5,
      mode="independent",
      adaptive_alpha=1.0,
      adaptive_uniform_ratio=0.0,
      min_weight=0.0,
    ),
    device="cpu",
  )

  sampler.update_failures(
    pair_indices=torch.zeros(5, dtype=torch.long),
    local_frames=torch.tensor([4, 4, 4, 4, 0]),
    failure_mask=torch.ones(5, dtype=torch.bool),
  )

  torch.testing.assert_close(
    sampler.bin_weights[0],
    torch.tensor([0.2, 0.0, 0.8, 0.0, 0.0]),
  )


def test_pair_frame_sampler_adaptive_keeps_probability_floor() -> None:
  sampler = PairFrameSampler(
    frame_counts=torch.tensor([100]),
    cfg=PairFrameSamplerCfg(
      num_bins=10,
      mode="independent",
      adaptive_alpha=0.001,
      adaptive_uniform_ratio=0.1,
      min_weight=0.0,
    ),
    device="cpu",
  )

  pair_indices = torch.zeros(1000, dtype=torch.long)
  local_frames = torch.zeros(1000, dtype=torch.long)
  failure_mask = torch.ones(1000, dtype=torch.bool)
  for _ in range(100):
    sampler.update_failures(pair_indices, local_frames, failure_mask)

  top1 = float(sampler.bin_weights[0].max())
  assert top1 < 0.55
  assert float(sampler.bin_weights[0].min()) > 0.0
  torch.testing.assert_close(sampler.bin_weights.sum(dim=1), torch.ones(1))


def test_pair_frame_sampler_pair_distribution_stays_record_weighted() -> None:
  sampler = PairFrameSampler(
    frame_counts=torch.tensor([20, 20, 20]),
    cfg=PairFrameSamplerCfg(
      num_bins=5,
      mode="independent",
      adaptive_alpha=0.001,
      adaptive_uniform_ratio=0.1,
    ),
    device="cpu",
    pair_weights=torch.tensor([1.0, 2.0, 1.0]),
  )

  pair_indices = torch.zeros(500, dtype=torch.long)
  local_frames = torch.zeros(500, dtype=torch.long)
  failure_mask = torch.ones(500, dtype=torch.bool)
  for _ in range(50):
    sampler.update_failures(pair_indices, local_frames, failure_mask)

  torch.testing.assert_close(
    sampler.pair_probabilities,
    torch.tensor([0.25, 0.5, 0.25]),
  )


def test_pair_frame_sampler_joint_probabilities_match_pair_times_bin_probs() -> None:
  sampler = PairFrameSampler(
    frame_counts=torch.tensor([10, 10]),
    cfg=PairFrameSamplerCfg(num_bins=4, mode="independent"),
    device="cpu",
    pair_weights=torch.tensor([1.0, 3.0]),
  )

  joint = sampler.joint_bin_probabilities

  assert joint.shape == (2, 4)
  torch.testing.assert_close(joint.sum(), torch.tensor(1.0))
  torch.testing.assert_close(joint.sum(dim=1), sampler.pair_probabilities)


def test_pair_frame_sampler_concat_mode_samples_valid_pair_bins():
  sampler = PairFrameSampler(
    frame_counts=torch.tensor([10, 20, 30]),
    cfg=PairFrameSamplerCfg(num_bins=5, mode="concat_pair_bins"),
    device="cpu",
  )

  sample = sampler.sample(torch.arange(64))

  assert torch.all(sample.pair_indices >= 0)
  assert torch.all(sample.pair_indices < 3)
  assert torch.all(sample.local_frames >= 0)
  assert torch.all(sample.local_frames < sampler.frame_counts[sample.pair_indices])
  torch.testing.assert_close(
    sampler.concat_bin_weights.sum(),
    torch.tensor(1.0),
  )
