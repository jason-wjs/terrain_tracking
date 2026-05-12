from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from terrain_tracking.tasks.blind_terrain_tracking.config.g1.rl_cfg import (
  unitree_g1_blind_terrain_tracking_ppo_runner_cfg,
)


def unitree_g1_oracle_terrain_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_g1_blind_terrain_tracking_ppo_runner_cfg()
  cfg.experiment_name = "g1_oracle_terrain_tracking"
  return cfg
