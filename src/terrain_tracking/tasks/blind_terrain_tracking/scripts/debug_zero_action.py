from __future__ import annotations

from dataclasses import dataclass

import torch
import tyro
from mjlab.tasks.tracking.mdp import MotionCommandCfg

from terrain_tracking.tasks.blind_terrain_tracking.scripts.common import (
  DEFAULT_TASK_ID,
  build_paired_env,
  configure_runtime,
)


@dataclass(frozen=True)
class DebugConfig:
  pair_manifest: str
  task: str = DEFAULT_TASK_ID
  device: str | None = "cpu"
  num_envs: int = 64
  steps: int = 12
  collision_backend: str = "primitive_boxes"


def main() -> None:
  args = tyro.cli(DebugConfig)
  device = configure_runtime(args.device)
  env, _agent_cfg = build_paired_env(
    args.task,
    args.pair_manifest,
    play=False,
    device=device,
    num_envs=args.num_envs,
    no_terminations=True,
    collision_backend=args.collision_backend,
  )
  try:
    cmd = env.command_manager.get_term("motion")
    assert isinstance(cmd.cfg, MotionCommandCfg)
    cmd.cfg.sampling_mode = "start"
    body_names = list(cmd.cfg.body_names)
    ee_ids = [
      body_names.index("left_ankle_roll_link"),
      body_names.index("right_ankle_roll_link"),
      body_names.index("left_wrist_yaw_link"),
      body_names.index("right_wrist_yaw_link"),
    ]
    env.reset()
    zero = torch.zeros(env.action_space.shape, device=env.device)
    for step in range(args.steps + 1):
      err = torch.abs(
        cmd.body_pos_relative_w[:, ee_ids, 2] - cmd.robot_body_pos_w[:, ee_ids, 2]
      )
      anchor = torch.abs(cmd.anchor_pos_w[:, 2] - cmd.robot_anchor_pos_w[:, 2])
      print(
        f"state={step:02d} "
        f"max_ee_z={float(err.max()):.6f} "
        f"max_anchor_z={float(anchor.max()):.6f}"
      )
      if step < args.steps:
        env.step(zero)
  finally:
    env.close()


if __name__ == "__main__":
  main()
