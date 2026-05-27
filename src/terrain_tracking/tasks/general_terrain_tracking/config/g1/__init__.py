"""Task registration for Unitree G1 general oracle terrain tracking."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import unitree_g1_general_oracle_teacher_terrain_tracking_env_cfg
from .rl_cfg import unitree_g1_general_oracle_teacher_terrain_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="TT-Tracking-TerrainOracleTeacherGeneral-Unitree-G1",
  env_cfg=unitree_g1_general_oracle_teacher_terrain_tracking_env_cfg(),
  play_env_cfg=unitree_g1_general_oracle_teacher_terrain_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_general_oracle_teacher_terrain_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="TT-Tracking-TerrainOracleTeacherGeneral-Unitree-G1-Play",
  env_cfg=unitree_g1_general_oracle_teacher_terrain_tracking_env_cfg(play=True),
  play_env_cfg=unitree_g1_general_oracle_teacher_terrain_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_general_oracle_teacher_terrain_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
