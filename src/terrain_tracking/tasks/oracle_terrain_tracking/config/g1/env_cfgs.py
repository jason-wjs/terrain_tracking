from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg

from terrain_tracking.tasks.blind_terrain_tracking.config.g1.env_cfgs import (
  unitree_g1_blind_terrain_tracking_env_cfg,
)

TERRAIN_SCAN_SENSOR_NAME = "terrain_scan"
HEIGHT_SCAN_MAX_DISTANCE = 5.0


def _height_scan_term() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=envs_mdp.height_scan,
    params={"sensor_name": TERRAIN_SCAN_SENSOR_NAME},
    scale=1.0 / HEIGHT_SCAN_MAX_DISTANCE,
  )


def _add_oracle_height_scan(cfg: ManagerBasedRlEnvCfg) -> None:
  terrain_scan = RayCastSensorCfg(
    name=TERRAIN_SCAN_SENSOR_NAME,
    frame=ObjRef(type="body", name="torso_link", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(0.7, 0.7), resolution=0.1),
    max_distance=HEIGHT_SCAN_MAX_DISTANCE,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (terrain_scan,)

  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms["height_scan"] = _height_scan_term()


def unitree_g1_oracle_height_terrain_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_blind_terrain_tracking_env_cfg(play=play)
  _add_oracle_height_scan(cfg)
  return cfg


def unitree_g1_oracle_teacher_terrain_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return unitree_g1_oracle_height_terrain_tracking_env_cfg(play=play)
