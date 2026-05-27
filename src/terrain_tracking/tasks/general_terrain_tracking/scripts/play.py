from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import mjlab
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from terrain_tracking.runtime.apply_pair_dataset import apply_pair_dataset_to_env_cfg
from terrain_tracking.runtime.pair_dataset import ValidationMode
from terrain_tracking.runtime.pair_frame_sampler import PairFrameSamplerCfg, SamplerMode
from terrain_tracking.tasks.blind_terrain_tracking.scripts.common import (
  build_policy,
  configure_runtime,
)
from terrain_tracking.tasks.general_terrain_tracking.mdp import MultiMotionCommandCfg

DEFAULT_TASK_ID = "TT-Tracking-TerrainOracleTeacherGeneral-Unitree-G1"


@dataclass(frozen=True)
class PlayConfig:
  pair_dataset: str
  task: str = DEFAULT_TASK_ID
  agent: Literal["zero", "random", "trained"] = "zero"
  checkpoint_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  dataset_validate: ValidationMode = "fast"
  max_pairs: int | None = None
  pair_filter: tuple[str, ...] = ()
  pair_sampler_mode: SamplerMode = "independent"


def _apply_frontend(env_cfg, args: PlayConfig) -> None:
  motion_cmd = env_cfg.commands["motion"]
  if not isinstance(motion_cmd, MultiMotionCommandCfg):
    raise TypeError("Expected cfg.commands['motion'] to be a MultiMotionCommandCfg")
  motion_cmd.pair_dataset = args.pair_dataset
  motion_cmd.dataset_validate = args.dataset_validate
  motion_cmd.max_pairs = args.max_pairs
  motion_cmd.pair_filter = args.pair_filter
  motion_cmd.sampler = PairFrameSamplerCfg(
    num_bins=motion_cmd.sampler.num_bins,
    mode=args.pair_sampler_mode,
    ema_alpha=motion_cmd.sampler.ema_alpha,
    min_weight=motion_cmd.sampler.min_weight,
  )
  apply_pair_dataset_to_env_cfg(env_cfg)


def main() -> None:
  args = tyro.cli(PlayConfig, config=mjlab.TYRO_FLAGS)
  device = configure_runtime(args.device)
  env_cfg = load_env_cfg(args.task, play=True)
  agent_cfg = load_rl_cfg(args.task)
  if args.num_envs is not None:
    env_cfg.scene.num_envs = args.num_envs
  if args.no_terminations:
    env_cfg.terminations = {}
  _apply_frontend(env_cfg, args)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
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
