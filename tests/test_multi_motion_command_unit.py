from __future__ import annotations

import torch

from terrain_tracking.runtime.pair_frame_sampler import PairFrameSamplerCfg
from terrain_tracking.tasks.general_terrain_tracking.mdp import observations
from terrain_tracking.tasks.general_terrain_tracking.mdp.multi_motion_command import (
  MultiMotionCommandCfg,
)


def test_multi_motion_command_cfg_exposes_general_dataset_fields() -> None:
  cfg = MultiMotionCommandCfg(
    pair_dataset="/tmp/pairs.jsonl",
    entity_name="robot",
    resampling_time_range=(1.0e9, 1.0e9),
    anchor_body_name="pelvis",
    body_names=("pelvis", "left_foot"),
  )

  assert cfg.pair_dataset == "/tmp/pairs.jsonl"
  assert cfg.dataset_validate == "fast"
  assert cfg.sampler.mode == "independent"
  assert isinstance(cfg.sampler, PairFrameSamplerCfg)
  assert cfg.sampler.adaptive_alpha == 1.0e-3
  assert cfg.sampler.adaptive_uniform_ratio == 0.1
  assert not hasattr(cfg.sampler, "ema_alpha")


def test_pelvis_pair_local_pos_observation_removes_tile_origin() -> None:
  class _FakeCommandCfg:
    body_names = ("pelvis", "left_foot")

  class _FakeCommand:
    cfg = _FakeCommandCfg()
    robot_body_pos_pair = torch.tensor(
      [
        [[0.7, 0.2, 0.8], [0.9, 0.1, 0.0]],
        [[1.5, -0.4, 0.9], [1.8, -0.5, 0.0]],
      ],
      dtype=torch.float32,
    )

  class _FakeCommandManager:
    def get_term(self, command_name: str) -> _FakeCommand:
      assert command_name == "motion"
      return _FakeCommand()

  class _FakeEnv:
    num_envs = 2
    command_manager = _FakeCommandManager()

  result = observations.pelvis_pair_local_pos_w(_FakeEnv(), "motion")

  torch.testing.assert_close(
    result,
    torch.tensor(
      [
        [0.7, 0.2, 0.8],
        [1.5, -0.4, 0.9],
      ],
      dtype=torch.float32,
    ),
  )
