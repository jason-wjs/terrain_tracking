from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from terrain_tracking.runtime.apply_pair import apply_pair_manifest_to_env_cfg

DEFAULT_TASK_ID = "TT-Tracking-TerrainBlind-Unitree-G1"
PolicyMode = Literal["zero", "random", "trained"]


def import_local_tasks() -> None:
  import terrain_tracking._mjlab_tasks  # noqa: F401


def resolve_device(device: str | None) -> str:
  return device or ("cuda:0" if torch.cuda.is_available() else "cpu")


def build_paired_env(
  task_id: str,
  pair_manifest: str,
  *,
  play: bool,
  device: str,
  num_envs: int | None = None,
  env_spacing: float | None = None,
  render_mode: str | None = None,
  no_terminations: bool = False,
  collision_backend: str = "primitive_boxes",
) -> tuple[ManagerBasedRlEnv, RslRlBaseRunnerCfg]:
  env_cfg = load_env_cfg(task_id, play=play)
  agent_cfg = load_rl_cfg(task_id)

  if num_envs is not None:
    env_cfg.scene.num_envs = num_envs
  if env_spacing is not None:
    env_cfg.scene.env_spacing = env_spacing
  apply_pair_manifest_to_env_cfg(
    env_cfg,
    pair_manifest,
    collision_backend=collision_backend,
  )
  if no_terminations:
    env_cfg.terminations = {}

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
  return env, agent_cfg


def build_policy(
  task_id: str,
  env: RslRlVecEnvWrapper,
  agent_cfg: RslRlBaseRunnerCfg,
  *,
  agent: PolicyMode,
  device: str,
  checkpoint_file: str | None = None,
):
  if agent == "zero":
    action_shape = env.unwrapped.action_space.shape

    class PolicyZero:
      def __call__(self, obs) -> torch.Tensor:
        del obs
        return torch.zeros(action_shape, device=env.unwrapped.device)

    return PolicyZero()

  if agent == "random":
    action_shape = env.unwrapped.action_space.shape

    class PolicyRandom:
      def __call__(self, obs) -> torch.Tensor:
        del obs
        return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

    return PolicyRandom()

  if checkpoint_file is None:
    raise ValueError("checkpoint_file is required when agent='trained'")

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(Path(checkpoint_file)),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  return runner.get_inference_policy(device=device)


def configure_runtime(device: str | None) -> str:
  configure_torch_backends()
  import_local_tasks()
  return resolve_device(device)
