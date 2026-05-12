from __future__ import annotations

import torch
from mjlab.envs import mdp as envs_mdp
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg

from terrain_tracking.tasks.blind_terrain_tracking.config.g1.env_cfgs import (
  unitree_g1_blind_terrain_tracking_env_cfg,
)
from terrain_tracking.tasks.oracle_terrain_tracking.config.g1 import (
  observations as oracle_obs,
)
from terrain_tracking.tasks.oracle_terrain_tracking.config.g1.env_cfgs import (
  unitree_g1_oracle_height_terrain_tracking_env_cfg,
  unitree_g1_oracle_teacher_terrain_tracking_env_cfg,
)

ORACLE_TEACHER_TERMS = {
  "global_anchor_pos_error_w",
  "global_anchor_lin_vel_error_w",
  "reference_anchor_lin_vel_w",
}


def _term_names(cfg, group_name: str) -> set[str]:
  return set(cfg.observations[group_name].terms)


def test_blind_config_does_not_include_oracle_observations() -> None:
  cfg = unitree_g1_blind_terrain_tracking_env_cfg()

  for group_name in ("actor", "critic"):
    term_names = _term_names(cfg, group_name)
    assert "height_scan" not in term_names
    assert ORACLE_TEACHER_TERMS.isdisjoint(term_names)

  sensor_names = {sensor.name for sensor in cfg.scene.sensors or ()}
  assert "terrain_scan" not in sensor_names


def test_oracle_height_adds_clean_height_scan_to_actor_and_critic() -> None:
  cfg = unitree_g1_oracle_height_terrain_tracking_env_cfg()

  for group_name in ("actor", "critic"):
    term = cfg.observations[group_name].terms["height_scan"]
    assert term.func is envs_mdp.height_scan
    assert term.params == {"sensor_name": "terrain_scan"}
    assert term.noise is None
    assert term.scale == 0.2
    assert term.delay_min_lag == 0
    assert term.delay_max_lag == 0

  sensor_by_name = {sensor.name: sensor for sensor in cfg.scene.sensors or ()}
  terrain_scan = sensor_by_name["terrain_scan"]
  assert isinstance(terrain_scan, RayCastSensorCfg)
  assert isinstance(terrain_scan.frame, ObjRef)
  assert terrain_scan.frame.type == "body"
  assert terrain_scan.frame.name == "torso_link"
  assert terrain_scan.frame.entity == "robot"
  assert terrain_scan.ray_alignment == "yaw"
  assert terrain_scan.max_distance == 5.0
  assert terrain_scan.include_geom_groups == (0,)
  assert isinstance(terrain_scan.pattern, GridPatternCfg)
  assert terrain_scan.pattern.size == (0.7, 0.7)
  assert terrain_scan.pattern.resolution == 0.1


def test_oracle_height_grid_pattern_has_expected_current_ray_count() -> None:
  cfg = unitree_g1_oracle_height_terrain_tracking_env_cfg()
  sensor_by_name = {sensor.name: sensor for sensor in cfg.scene.sensors or ()}
  terrain_scan = sensor_by_name["terrain_scan"]
  assert isinstance(terrain_scan, RayCastSensorCfg)
  assert isinstance(terrain_scan.pattern, GridPatternCfg)

  offsets, directions = terrain_scan.pattern.generate_rays(None, "cpu")

  assert offsets.shape == (64, 3)
  assert directions.shape == (64, 3)
  assert torch.allclose(directions, torch.tensor([[0.0, 0.0, -1.0]]).repeat(64, 1))


def test_oracle_teacher_includes_height_scan() -> None:
  cfg = unitree_g1_oracle_teacher_terrain_tracking_env_cfg()

  for group_name in ("actor", "critic"):
    term_names = _term_names(cfg, group_name)
    assert "height_scan" in term_names


class _FakeMotionCommand:
  def __init__(self) -> None:
    self.anchor_pos_w = torch.tensor(
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      dtype=torch.float32,
    )
    self.robot_anchor_pos_w = torch.tensor(
      [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]],
      dtype=torch.float32,
    )
    self.anchor_lin_vel_w = torch.tensor(
      [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
      dtype=torch.float32,
    )
    self.robot_anchor_lin_vel_w = torch.tensor(
      [[0.0, 0.1, 0.1], [0.1, 0.2, 0.3]],
      dtype=torch.float32,
    )


class _FakeCommandManager:
  def __init__(self, command: _FakeMotionCommand) -> None:
    self.command = command

  def get_term(self, command_name: str) -> _FakeMotionCommand:
    assert command_name == "motion"
    return self.command


class _FakeEnv:
  def __init__(self, command: _FakeMotionCommand) -> None:
    self.command_manager = _FakeCommandManager(command)


def test_teacher_privileged_observation_functions_return_expected_values() -> None:
  command = _FakeMotionCommand()
  env = _FakeEnv(command)

  assert torch.allclose(
    oracle_obs.global_anchor_pos_error_w(env, "motion"),
    torch.full((2, 3), 0.5),
  )
  assert torch.allclose(
    oracle_obs.global_anchor_lin_vel_error_w(env, "motion"),
    torch.tensor([[0.1, 0.1, 0.2], [0.3, 0.3, 0.3]], dtype=torch.float32),
  )
  assert torch.allclose(
    oracle_obs.reference_anchor_lin_vel_w(env, "motion"),
    command.anchor_lin_vel_w,
  )


def test_oracle_height_does_not_include_teacher_privileged_terms() -> None:
  cfg = unitree_g1_oracle_height_terrain_tracking_env_cfg()

  for group_name in ("actor", "critic"):
    assert ORACLE_TEACHER_TERMS.isdisjoint(_term_names(cfg, group_name))


def test_oracle_teacher_adds_privileged_terms_to_actor_and_critic() -> None:
  cfg = unitree_g1_oracle_teacher_terrain_tracking_env_cfg()

  expected_funcs = {
    "global_anchor_pos_error_w": oracle_obs.global_anchor_pos_error_w,
    "global_anchor_lin_vel_error_w": oracle_obs.global_anchor_lin_vel_error_w,
    "reference_anchor_lin_vel_w": oracle_obs.reference_anchor_lin_vel_w,
  }
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    assert "height_scan" in terms
    for term_name, func in expected_funcs.items():
      term = terms[term_name]
      assert term.func is func
      assert term.params == {"command_name": "motion"}
      assert term.noise is None
