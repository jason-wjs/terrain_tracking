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
    sampling_metric_keys = {
      "sampling_entropy",
      "sampling_top1_prob",
      "sampling_pair_entropy",
      "sampling_pair_top1_prob",
      "sampling_bin_entropy_mean",
      "sampling_bin_entropy_min",
      "sampling_bin_top1_prob_mean",
      "sampling_bin_top1_prob_max",
      "active_pair_entropy",
      "active_pair_top1_frac",
    }
    assert sampling_metric_keys.issubset(command.metrics)
    tracking_metric_keys = {
      "error_anchor_pos",
      "error_anchor_rot",
      "error_anchor_lin_vel",
      "error_anchor_ang_vel",
      "error_body_pos",
      "error_body_rot",
      "error_body_lin_vel",
      "error_body_ang_vel",
      "error_joint_pos",
      "error_joint_vel",
    }
    assert tracking_metric_keys.issubset(command.metrics)
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
    for key in sampling_metric_keys:
      value = command.metrics[key]
      assert value.shape == (env.num_envs,)
      assert torch.isfinite(value).all()
    for key in tracking_metric_keys:
      value = command.metrics[key]
      assert value.shape == (env.num_envs,)
      assert torch.isfinite(value).all()
  finally:
    env.close()


def test_general_start_sampling_uses_multiple_pairs(tmp_path: Path) -> None:
  root = tmp_path / "parc"
  _create_pair(root, "platform/a", frames=8)
  _create_pair(root, "stairs/b", frames=9)
  manifest = tmp_path / "pair_dataset.jsonl"
  build_parc_pair_dataset(BuildPairDatasetConfig(root=root, output=manifest))

  cfg = unitree_g1_general_oracle_teacher_terrain_tracking_env_cfg(
    pair_dataset=str(manifest),
    play=True,
  )
  cfg.scene.num_envs = 16
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MultiMotionCommandCfg)
  motion_cmd.sampling_mode = "start"

  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset()
    command = env.command_manager.get_term("motion")
    assert torch.unique(command.env_pair_indices).numel() > 1
    assert torch.all(command.time_steps <= 1)
  finally:
    env.close()


def test_general_pair_dataset_uses_fixed_contact_buffer_defaults(
  tmp_path: Path,
) -> None:
  root = tmp_path / "parc"
  for pair_index in range(40):
    _create_pair(root, f"platform/pair_{pair_index:03d}", frames=8)
  manifest = tmp_path / "pair_dataset.jsonl"
  build_parc_pair_dataset(BuildPairDatasetConfig(root=root, output=manifest))

  cfg = unitree_g1_general_oracle_teacher_terrain_tracking_env_cfg(
    pair_dataset=str(manifest),
    play=True,
  )

  assert cfg.sim.nconmax == 256
  assert cfg.sim.njmax == 512
