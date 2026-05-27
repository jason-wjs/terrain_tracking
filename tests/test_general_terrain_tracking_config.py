from __future__ import annotations

from mjlab.managers.termination_manager import TerminationTermCfg

from terrain_tracking.tasks.general_terrain_tracking.config.g1.env_cfgs import (
  unitree_g1_general_oracle_teacher_terrain_tracking_env_cfg,
)
from terrain_tracking.tasks.general_terrain_tracking.mdp import (
  MultiMotionCommandCfg,
  out_of_tile_bounds,
)


def test_general_teacher_cfg_replaces_motion_command_and_adds_tile_termination():
  cfg = unitree_g1_general_oracle_teacher_terrain_tracking_env_cfg()

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MultiMotionCommandCfg)
  assert motion_cmd.pair_dataset == ""
  assert motion_cmd.body_names
  assert motion_cmd.anchor_body_name

  assert "height_scan" in cfg.observations["actor"].terms
  assert "height_scan" in cfg.observations["critic"].terms
  termination = cfg.terminations["out_of_tile_bounds"]
  assert isinstance(termination, TerminationTermCfg)
  assert termination.func is out_of_tile_bounds
  assert termination.time_out is False
