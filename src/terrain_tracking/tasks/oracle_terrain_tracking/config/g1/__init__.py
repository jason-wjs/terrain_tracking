"""Task registration for Unitree G1 oracle terrain tracking."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  unitree_g1_oracle_height_terrain_tracking_env_cfg,
  unitree_g1_oracle_height_long_scan_php_reward_terrain_tracking_env_cfg,
  unitree_g1_oracle_teacher_terrain_tracking_env_cfg,
)
from .rl_cfg import unitree_g1_oracle_terrain_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="TT-Tracking-TerrainOracleHeight-Unitree-G1",
  env_cfg=unitree_g1_oracle_height_terrain_tracking_env_cfg(),
  play_env_cfg=unitree_g1_oracle_height_terrain_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_oracle_terrain_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="TT-Tracking-TerrainOracleHeightLongScanPhpReward-Unitree-G1",
  env_cfg=unitree_g1_oracle_height_long_scan_php_reward_terrain_tracking_env_cfg(),
  play_env_cfg=unitree_g1_oracle_height_long_scan_php_reward_terrain_tracking_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_oracle_terrain_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="TT-Tracking-TerrainOracleTeacher-Unitree-G1",
  env_cfg=unitree_g1_oracle_teacher_terrain_tracking_env_cfg(),
  play_env_cfg=unitree_g1_oracle_teacher_terrain_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_oracle_terrain_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
