from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from mjlab.envs import ManagerBasedRlEnv

from terrain_tracking.build_pair_dataset import (
  BuildPairDatasetConfig,
  build_parc_pair_dataset,
)
from terrain_tracking.tasks.general_terrain_tracking.config.g1.env_cfgs import (
  unitree_g1_general_oracle_teacher_terrain_tracking_env_cfg,
)
from terrain_tracking.tasks.general_terrain_tracking.mdp import MultiMotionCommandCfg
from tests.helpers import create_heightfield_collision_manifest, create_motion_clip


def _create_pair(root: Path, relative: str, *, frames: int) -> None:
  pair_dir = root / relative
  pair_dir.mkdir(parents=True)
  create_motion_clip(pair_dir / "motion.npz", num_frames=frames)
  create_heightfield_collision_manifest(pair_dir / "terrain_collision.json")


def test_general_oracle_teacher_env_can_reset_and_step_on_cpu(tmp_path: Path) -> None:
  root = tmp_path / "parc"
  _create_pair(root, "platform/a", frames=8)
  _create_pair(root, "stairs/b", frames=9)
  manifest = tmp_path / "pair_dataset.jsonl"
  build_parc_pair_dataset(BuildPairDatasetConfig(root=root, output=manifest))

  cfg = unitree_g1_general_oracle_teacher_terrain_tracking_env_cfg(
    pair_dataset=str(manifest),
    play=True,
  )
  cfg.scene.num_envs = 1
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MultiMotionCommandCfg)
  motion_cmd.sampling_mode = "start"

  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    obs, _extras = env.reset()
    assert set(obs.keys()) == {"actor", "critic"}
    command = env.command_manager.get_term("motion")
    assert torch.allclose(
      command.anchor_pos_w,
      command.body_pos_w[:, command.motion_anchor_body_index],
    )
    assert torch.allclose(
      command.anchor_quat_w,
      command.body_quat_w[:, command.motion_anchor_body_index],
    )

    action_dim = env.unwrapped.single_action_space.shape[0]
    action = torch.zeros((1, action_dim), dtype=torch.float32, device=env.device)
    obs, reward, terminated, timeouts, extras = env.step(action)
    actor_obs = cast(torch.Tensor, obs["actor"])
    assert actor_obs.shape[0] == 1
    assert reward.shape == (1,)
    assert terminated.shape == (1,)
    assert timeouts.shape == (1,)
    assert isinstance(extras, dict)
  finally:
    env.close()
