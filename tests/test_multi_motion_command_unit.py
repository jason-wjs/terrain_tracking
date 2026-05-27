from __future__ import annotations

from terrain_tracking.runtime.pair_frame_sampler import PairFrameSamplerCfg
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
