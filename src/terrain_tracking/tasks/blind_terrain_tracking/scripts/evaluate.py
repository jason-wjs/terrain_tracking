from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mjlab
import torch
import tyro
from mjlab.rl import RslRlVecEnvWrapper

from terrain_tracking.tasks.blind_terrain_tracking.scripts.common import (
  DEFAULT_TASK_ID,
  build_paired_env,
  build_policy,
  configure_runtime,
)


@dataclass(frozen=True)
class EvaluateConfig:
  pair_manifest: str
  task: str = DEFAULT_TASK_ID
  agent: Literal["zero", "random", "trained"] = "zero"
  checkpoint_file: str | None = None
  num_envs: int = 1
  device: str | None = None
  num_steps: int = 200
  no_terminations: bool = False


def main() -> None:
  args = tyro.cli(EvaluateConfig, config=mjlab.TYRO_FLAGS)
  device = configure_runtime(args.device)

  env, agent_cfg = build_paired_env(
    args.task,
    args.pair_manifest,
    play=False,
    device=device,
    num_envs=args.num_envs,
    no_terminations=args.no_terminations,
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

  rewards: list[torch.Tensor] = []
  obs = vec_env.get_observations()
  for _ in range(args.num_steps):
    with torch.no_grad():
      actions = policy(obs)
    obs, reward, _dones, _extras = vec_env.step(actions)
    rewards.append(reward)

  mean_reward = torch.stack(rewards).mean().item()
  print(f"mean_reward={mean_reward:.6f}")
  vec_env.close()


if __name__ == "__main__":
  main()
