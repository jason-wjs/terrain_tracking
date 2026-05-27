from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

from terrain_tracking.runtime.apply_pair_dataset import apply_pair_dataset_to_env_cfg
from terrain_tracking.tasks.general_terrain_tracking import mdp
from terrain_tracking.tasks.general_terrain_tracking.mdp import MultiMotionCommandCfg
from terrain_tracking.tasks.oracle_terrain_tracking.config.g1.env_cfgs import (
  unitree_g1_oracle_teacher_terrain_tracking_env_cfg,
)


def unitree_g1_general_oracle_teacher_terrain_tracking_env_cfg(
  *,
  pair_dataset: str = "",
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_oracle_teacher_terrain_tracking_env_cfg(play=play)
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)

  cfg.commands["motion"] = MultiMotionCommandCfg(
    pair_dataset=pair_dataset,
    entity_name=motion_cmd.entity_name,
    resampling_time_range=motion_cmd.resampling_time_range,
    debug_vis=motion_cmd.debug_vis,
    anchor_body_name=motion_cmd.anchor_body_name,
    body_names=motion_cmd.body_names,
    pose_range=motion_cmd.pose_range,
    velocity_range=motion_cmd.velocity_range,
    joint_position_range=motion_cmd.joint_position_range,
    sampling_mode=motion_cmd.sampling_mode,
  )
  cfg.terminations["out_of_tile_bounds"] = TerminationTermCfg(
    func=mdp.out_of_tile_bounds,
    params={"command_name": "motion", "fail_margin": 1.0},
    time_out=False,
  )
  if pair_dataset:
    apply_pair_dataset_to_env_cfg(cfg)
  return cfg


__all__ = ["unitree_g1_general_oracle_teacher_terrain_tracking_env_cfg"]
