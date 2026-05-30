from __future__ import annotations

from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import GridPatternCfg, RayCastSensorCfg

from terrain_tracking.tasks.general_terrain_tracking.config.g1.env_cfgs import (
  unitree_g1_general_oracle_height_long_scan_php_reward_terrain_tracking_env_cfg,
  unitree_g1_general_oracle_teacher_terrain_tracking_env_cfg,
)
from terrain_tracking.tasks.general_terrain_tracking.mdp import (
  MultiMotionCommandCfg,
  observations,
  out_of_tile_bounds,
)
from terrain_tracking.tasks.oracle_terrain_tracking.config.g1 import (
  observations as oracle_observations,
)
from terrain_tracking.tasks.oracle_terrain_tracking.config.g1.env_cfgs import (
  unitree_g1_oracle_teacher_terrain_tracking_env_cfg,
)

ORACLE_TEACHER_TERMS = {
  "reference_pelvis_pos_error_b",
  "reference_pelvis_ori_error_b",
  "pelvis_lin_vel",
  "pelvis_ang_vel",
  "pelvis_global_pos_w",
  "pelvis_global_lin_vel_w",
}


def _terrain_scan(cfg):
  sensor_by_name = {sensor.name: sensor for sensor in cfg.scene.sensors or ()}
  return sensor_by_name["terrain_scan"]


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


def test_general_teacher_uses_pair_local_pelvis_position_observation() -> None:
  cfg = unitree_g1_general_oracle_teacher_terrain_tracking_env_cfg()

  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    assert "pelvis_global_pos_w" not in terms
    assert "pelvis_pair_local_pos_w" in terms
    assert terms["pelvis_pair_local_pos_w"].func is observations.pelvis_pair_local_pos_w


def test_single_oracle_teacher_keeps_world_pelvis_position_observation() -> None:
  cfg = unitree_g1_oracle_teacher_terrain_tracking_env_cfg()

  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    assert "pelvis_global_pos_w" in terms
    assert terms["pelvis_global_pos_w"].func is oracle_observations.pelvis_global_pos_w
    assert "pelvis_pair_local_pos_w" not in terms


def test_general_oracle_height_long_scan_php_reward_cfg_preserves_base_semantics():
  cfg = unitree_g1_general_oracle_height_long_scan_php_reward_terrain_tracking_env_cfg()

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MultiMotionCommandCfg)
  assert motion_cmd.pair_dataset == ""
  assert motion_cmd.body_names
  assert motion_cmd.anchor_body_name

  terrain_scan = _terrain_scan(cfg)
  assert isinstance(terrain_scan, RayCastSensorCfg)
  assert isinstance(terrain_scan.pattern, GridPatternCfg)
  assert terrain_scan.pattern.size == (2.0, 0.7)
  assert terrain_scan.pattern.resolution == 0.1

  assert cfg.rewards["motion_global_root_pos"].weight == 1.0
  assert cfg.rewards["motion_global_root_ori"].weight == 1.0
  assert cfg.rewards["self_collisions"].weight == -0.5

  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    assert "height_scan" in terms
    assert ORACLE_TEACHER_TERMS.isdisjoint(terms)
    assert "pelvis_pair_local_pos_w" not in terms

  termination = cfg.terminations["out_of_tile_bounds"]
  assert isinstance(termination, TerminationTermCfg)
  assert termination.func is out_of_tile_bounds
  assert termination.time_out is False
