from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import mjlab
import tyro
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from terrain_tracking.tasks.blind_terrain_tracking.scripts.common import (
  DEFAULT_TASK_ID,
  build_paired_env,
  build_policy,
  configure_runtime,
)


@dataclass(frozen=True)
class PlayConfig:
  pair_manifest: str
  task: str = DEFAULT_TASK_ID
  agent: Literal["zero", "random", "trained"] = "zero"
  checkpoint_file: str | None = None
  num_envs: int | None = None
  env_spacing: float | None = None
  device: str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  collision_backend: str = "primitive_boxes"


def main() -> None:
  args = tyro.cli(PlayConfig, config=mjlab.TYRO_FLAGS)
  device = configure_runtime(args.device)

  env, agent_cfg = build_paired_env(
    args.task,
    args.pair_manifest,
    play=True,
    device=device,
    num_envs=args.num_envs,
    env_spacing=args.env_spacing,
    no_terminations=args.no_terminations,
    collision_backend=args.collision_backend,
  )
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  policy = build_policy(
    args.task,
    vec_env,
    agent_cfg,
    agent=args.agent,
    device=device,
    checkpoint_file=args.checkpoint_file,
  )

  if args.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    viewer = "native" if has_display else "viser"
  else:
    viewer = args.viewer

  if viewer == "native":
    NativeMujocoViewer(vec_env, policy).run()
  else:
    ViserPlayViewer(vec_env, policy).run()

  vec_env.close()


if __name__ == "__main__":
  main()
