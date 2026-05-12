from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from terrain_tracking.tasks.blind_terrain_tracking.config.g1.env_cfgs import (
  unitree_g1_blind_terrain_tracking_env_cfg,
)


def unitree_g1_oracle_height_terrain_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return unitree_g1_blind_terrain_tracking_env_cfg(play=play)


def unitree_g1_oracle_teacher_terrain_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return unitree_g1_oracle_height_terrain_tracking_env_cfg(play=play)
