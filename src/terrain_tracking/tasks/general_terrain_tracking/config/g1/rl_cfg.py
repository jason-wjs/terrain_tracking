from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from terrain_tracking.tasks.oracle_terrain_tracking.config.g1.rl_cfg import (
  unitree_g1_oracle_terrain_tracking_ppo_runner_cfg,
)


def unitree_g1_general_oracle_teacher_terrain_tracking_ppo_runner_cfg() -> (
  RslRlOnPolicyRunnerCfg
):
  cfg = unitree_g1_oracle_terrain_tracking_ppo_runner_cfg()
  cfg.experiment_name = "g1_general_oracle_terrain_tracking"
  return cfg


__all__ = ["unitree_g1_general_oracle_teacher_terrain_tracking_ppo_runner_cfg"]
