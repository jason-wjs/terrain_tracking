from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg

from terrain_tracking.tasks.blind_terrain_tracking.config.g1.env_cfgs import (
  unitree_g1_blind_terrain_tracking_env_cfg,
)
from terrain_tracking.tasks.oracle_terrain_tracking.config.g1 import observations

TERRAIN_SCAN_SENSOR_NAME = "terrain_scan"
HEIGHT_SCAN_MAX_DISTANCE = 5.0


def _height_scan_term() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=envs_mdp.height_scan,
    params={"sensor_name": TERRAIN_SCAN_SENSOR_NAME},
    scale=1.0 / HEIGHT_SCAN_MAX_DISTANCE,
  )


def _teacher_term(func: Callable[..., Any]) -> ObservationTermCfg:
  return ObservationTermCfg(func=func, params={"command_name": "motion"})


def _add_oracle_height_scan(cfg: ManagerBasedRlEnvCfg) -> None:
  terrain_scan = RayCastSensorCfg(
    name=TERRAIN_SCAN_SENSOR_NAME,
    frame=ObjRef(type="body", name="torso_link", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(0.7, 0.7), resolution=0.1),
    max_distance=HEIGHT_SCAN_MAX_DISTANCE,
    exclude_parent_body=True,
    include_geom_groups=(0,),
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (terrain_scan,)

  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms["height_scan"] = _height_scan_term()


def _add_oracle_teacher_terms(cfg: ManagerBasedRlEnvCfg) -> None:
  teacher_terms = {
    "reference_pelvis_pos_error_b": observations.reference_pelvis_pos_error_b,
    "reference_pelvis_ori_error_b": observations.reference_pelvis_ori_error_b,
    "pelvis_lin_vel": observations.pelvis_lin_vel,
    "pelvis_ang_vel": observations.pelvis_ang_vel,
    "pelvis_global_pos_w": observations.pelvis_global_pos_w,
    "pelvis_global_lin_vel_w": observations.pelvis_global_lin_vel_w,
  }
  replaced_terms = {
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "body_pos",
    "body_ori",
    "base_lin_vel",
    "base_ang_vel",
    "global_anchor_pos_error_w",
    "global_anchor_lin_vel_error_w",
    "reference_anchor_lin_vel_w",
  }
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    for term_name in replaced_terms:
      terms.pop(term_name, None)
    for term_name, func in teacher_terms.items():
      terms[term_name] = _teacher_term(func)


def _align_oracle_teacher_rewards_with_php(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.rewards["motion_global_root_pos"].weight = 1.0
  cfg.rewards["motion_global_root_ori"].weight = 1.0
  cfg.rewards["self_collisions"].weight = -0.5


def unitree_g1_oracle_height_terrain_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_blind_terrain_tracking_env_cfg(play=play)
  _add_oracle_height_scan(cfg)
  return cfg


def unitree_g1_oracle_teacher_terrain_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_oracle_height_terrain_tracking_env_cfg(play=play)
  _align_oracle_teacher_rewards_with_php(cfg)
  _add_oracle_teacher_terms(cfg)
  return cfg
