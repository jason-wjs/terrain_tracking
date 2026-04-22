from __future__ import annotations

from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.tracking.mdp import MotionCommandCfg

from terrain_tracking.convert_pair import (
  ConvertPairConfig,
  convert_pair,
)
from terrain_tracking.runtime.apply_pair import apply_pair_manifest_to_env_cfg
from terrain_tracking.tasks.blind_terrain_tracking.config.g1.env_cfgs import (
  unitree_g1_blind_terrain_tracking_env_cfg,
)
from terrain_tracking.tasks.blind_terrain_tracking.scripts.common import (
  build_paired_env,
)
from tests.helpers import create_motion_clip, create_pair_manifest, create_ramp_obj


def test_apply_pair_manifest_to_env_cfg_sets_motion_file_and_scene_spec_fn(
  tmp_path: Path,
) -> None:
  motion_path = create_motion_clip(tmp_path / "motion.npz")
  terrain_path = create_ramp_obj(tmp_path / "terrain.obj")
  manifest_path = create_pair_manifest(
    tmp_path / "pair.json",
    motion_file=motion_path.name,
    terrain_file=terrain_path.name,
  )

  cfg = unitree_g1_blind_terrain_tracking_env_cfg()
  returned_manifest = apply_pair_manifest_to_env_cfg(cfg, manifest_path)

  motion_cfg = cfg.commands["motion"]
  assert isinstance(motion_cfg, MotionCommandCfg)
  assert motion_cfg.motion_file == str(motion_path.resolve())
  assert returned_manifest.motion_file == motion_path.resolve()
  assert cfg.scene.spec_fn is not None


def test_blind_terrain_tracking_env_can_reset_and_step_on_cpu(
  tmp_path: Path,
) -> None:
  motion_path = create_motion_clip(tmp_path / "motion.npz")
  terrain_path = create_ramp_obj(tmp_path / "terrain.obj")
  manifest_path = create_pair_manifest(
    tmp_path / "pair.json",
    motion_file=motion_path.name,
    terrain_file=terrain_path.name,
  )

  cfg = unitree_g1_blind_terrain_tracking_env_cfg()
  cfg.scene.num_envs = 1
  apply_pair_manifest_to_env_cfg(cfg, manifest_path)

  motion_cfg = cfg.commands["motion"]
  assert isinstance(motion_cfg, MotionCommandCfg)
  motion_cfg.sampling_mode = "start"

  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  obs, _extras = env.reset()
  assert set(obs.keys()) == {"actor", "critic"}

  action_dim = env.unwrapped.single_action_space.shape[0]
  action = torch.zeros((1, action_dim), dtype=torch.float32, device=env.device)
  obs, reward, terminated, timeouts, extras = env.step(action)
  assert obs["actor"].shape[0] == 1
  assert reward.shape == (1,)
  assert terminated.shape == (1,)
  assert timeouts.shape == (1,)
  assert isinstance(extras, dict)
  env.close()


def test_converted_pair_bundle_loads(tmp_path: Path) -> None:
  bundle_dir = convert_pair(
    ConvertPairConfig(
      motion_file=create_motion_clip(tmp_path / "motion.npz"),
      terrain_file=create_ramp_obj(tmp_path / "terrain.obj"),
      output_dir=tmp_path / "converted",
      sample_name="smoke_pair",
    )
  )

  env, _agent_cfg = build_paired_env(
    "TT-Tracking-TerrainBlind-Unitree-G1",
    str(bundle_dir / "pair.json"),
    play=True,
    device="cpu",
    num_envs=1,
    no_terminations=True,
  )
  try:
    motion_cfg = env.cfg.commands["motion"]
    assert isinstance(motion_cfg, MotionCommandCfg)
    assert motion_cfg.motion_file.endswith("motion.npz")
    assert env.cfg.scene.spec_fn is not None
  finally:
    env.close()
