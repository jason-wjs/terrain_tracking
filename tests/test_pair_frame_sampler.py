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
      ema_alpha=1.0,
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
