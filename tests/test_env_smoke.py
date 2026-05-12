from __future__ import annotations

from pathlib import Path

import mujoco
import pytest
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
from tests.helpers import (
  create_heightfield_collision_manifest,
  create_motion_clip,
  create_pair_manifest,
  create_ramp_obj,
)


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
  assert cfg.sim.nconmax == 35
  assert cfg.sim.njmax == 250


def test_pair_manifest_scene_spec_uses_late_num_env_override(
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
  apply_pair_manifest_to_env_cfg(cfg, manifest_path)
  cfg.scene.num_envs = 3

  assert cfg.scene.spec_fn is not None
  spec = mujoco.MjSpec()
  cfg.scene.spec_fn(spec)

  paired_bodies = [
    body for body in spec.worldbody.bodies if body.name.startswith("paired_terrain_")
  ]
  assert len(paired_bodies) == 3


def test_pair_manifest_scene_spec_uses_hfield_when_collision_manifest_exists(
  tmp_path: Path,
) -> None:
  motion_path = create_motion_clip(tmp_path / "motion.npz")
  terrain_path = create_ramp_obj(tmp_path / "terrain.obj")
  collision_path = create_heightfield_collision_manifest(
    tmp_path / "terrain_collision.json"
  )
  manifest_path = create_pair_manifest(
    tmp_path / "pair.json",
    motion_file=motion_path.name,
    terrain_file=terrain_path.name,
  )
  payload = manifest_path.read_text(encoding="utf-8")
  manifest_path.write_text(
    payload[:-1] + f', "terrain_collision_file": "{collision_path.name}"' + "}",
    encoding="utf-8",
  )

  cfg = unitree_g1_blind_terrain_tracking_env_cfg()
  apply_pair_manifest_to_env_cfg(
    cfg,
    manifest_path,
    collision_backend="hfield",
  )
  cfg.scene.num_envs = 2

  assert cfg.scene.spec_fn is not None
  assert cfg.sim.nconmax == 256
  assert cfg.sim.njmax == 512
  spec = mujoco.MjSpec()
  cfg.scene.spec_fn(spec)

  paired_bodies = [
    body for body in spec.worldbody.bodies if body.name.startswith("paired_terrain_")
  ]
  assert len(spec.hfields) == 1
  assert len(paired_bodies) == 2
  assert all(body.geoms[0].type == mujoco.mjtGeom.mjGEOM_HFIELD for body in paired_bodies)


def test_hfield_pair_rejects_env_spacing_smaller_than_terrain_footprint(
  tmp_path: Path,
) -> None:
  motion_path = create_motion_clip(tmp_path / "motion.npz")
  terrain_path = create_ramp_obj(tmp_path / "terrain.obj")
  collision_path = create_heightfield_collision_manifest(
    tmp_path / "terrain_collision.json"
  )
  manifest_path = create_pair_manifest(
    tmp_path / "pair.json",
    motion_file=motion_path.name,
    terrain_file=terrain_path.name,
  )
  payload = manifest_path.read_text(encoding="utf-8")
  manifest_path.write_text(
    payload[:-1] + f', "terrain_collision_file": "{collision_path.name}"' + "}",
    encoding="utf-8",
  )

  cfg = unitree_g1_blind_terrain_tracking_env_cfg()
  apply_pair_manifest_to_env_cfg(
    cfg,
    manifest_path,
    collision_backend="hfield",
  )
  cfg.scene.num_envs = 2
  cfg.scene.env_spacing = 0.5

  assert cfg.scene.spec_fn is not None
  with pytest.raises(ValueError, match="env_spacing"):
    cfg.scene.spec_fn(mujoco.MjSpec())


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


def test_build_paired_env_applies_env_spacing_override(tmp_path: Path) -> None:
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
    env_spacing=3.5,
    no_terminations=True,
  )
  try:
    assert env.cfg.scene.env_spacing == 3.5
  finally:
    env.close()


def test_paired_env_disables_default_plane_collision(tmp_path: Path) -> None:
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
    model = env.unwrapped.sim.mj_model
    default_plane_id = mujoco.mj_name2id(
      model,
      mujoco.mjtObj.mjOBJ_GEOM,
      "terrain",
    )
    paired_mesh_id = mujoco.mj_name2id(
      model,
      mujoco.mjtObj.mjOBJ_GEOM,
      "paired_terrain_0",
    )

    assert default_plane_id >= 0
    assert paired_mesh_id >= 0
    assert model.geom_type[default_plane_id] == mujoco.mjtGeom.mjGEOM_PLANE
    assert model.geom_contype[default_plane_id] == 0
    assert model.geom_conaffinity[default_plane_id] == 0
    assert model.geom_type[paired_mesh_id] == mujoco.mjtGeom.mjGEOM_MESH
    assert model.geom_contype[paired_mesh_id] == 1
    assert model.geom_conaffinity[paired_mesh_id] == 1
  finally:
    env.close()


def test_paired_env_uses_hfield_collision_when_manifest_exists(tmp_path: Path) -> None:
  collision_path = create_heightfield_collision_manifest(
    tmp_path / "terrain_collision.json"
  )
  bundle_dir = convert_pair(
    ConvertPairConfig(
      motion_file=create_motion_clip(tmp_path / "motion.npz"),
      terrain_file=create_ramp_obj(tmp_path / "terrain.obj"),
      terrain_collision_file=collision_path,
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
    collision_backend="hfield",
  )
  try:
    model = env.unwrapped.sim.mj_model
    paired_terrain_id = mujoco.mj_name2id(
      model,
      mujoco.mjtObj.mjOBJ_GEOM,
      "paired_terrain_0",
    )

    assert paired_terrain_id >= 0
    assert model.geom_type[paired_terrain_id] == mujoco.mjtGeom.mjGEOM_HFIELD
    assert model.geom_contype[paired_terrain_id] == 1
    assert model.geom_conaffinity[paired_terrain_id] == 1
  finally:
    env.close()


def test_hfield_manifest_defaults_to_one_shared_primitive_box_tile(
  tmp_path: Path,
) -> None:
  collision_path = create_heightfield_collision_manifest(
    tmp_path / "terrain_collision.json"
  )
  bundle_dir = convert_pair(
    ConvertPairConfig(
      motion_file=create_motion_clip(tmp_path / "motion.npz"),
      terrain_file=create_ramp_obj(tmp_path / "terrain.obj"),
      terrain_collision_file=collision_path,
      output_dir=tmp_path / "converted",
      sample_name="smoke_pair",
    )
  )

  env, _agent_cfg = build_paired_env(
    "TT-Tracking-TerrainBlind-Unitree-G1",
    str(bundle_dir / "pair.json"),
    play=True,
    device="cpu",
    num_envs=4,
    no_terminations=True,
  )
  try:
    model = env.unwrapped.sim.mj_model
    terrain_geoms = [
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
      for geom_id in range(model.ngeom)
    ]
    primitive_terrain_geoms = [
      name for name in terrain_geoms if name and name.startswith("terrain_")
    ]

    assert model.nhfield == 0
    assert len(primitive_terrain_geoms) > 1
    assert len(primitive_terrain_geoms) < 10
    assert torch.allclose(
      env.unwrapped.scene.env_origins,
      torch.zeros_like(env.unwrapped.scene.env_origins),
    )
  finally:
    env.close()


@pytest.mark.parametrize(
  ("task_id", "min_extra_actor_dims"),
  [
    ("TT-Tracking-TerrainOracleHeight-Unitree-G1", 64),
    ("TT-Tracking-TerrainOracleTeacher-Unitree-G1", 73),
  ],
)
def test_oracle_terrain_tracking_env_can_reset_and_step_on_cpu(
  tmp_path: Path,
  task_id: str,
  min_extra_actor_dims: int,
) -> None:
  collision_path = create_heightfield_collision_manifest(
    tmp_path / "terrain_collision.json"
  )
  bundle_dir = convert_pair(
    ConvertPairConfig(
      motion_file=create_motion_clip(tmp_path / "motion.npz"),
      terrain_file=create_ramp_obj(tmp_path / "terrain.obj"),
      terrain_collision_file=collision_path,
      output_dir=tmp_path / "converted",
      sample_name="smoke_pair",
    )
  )

  blind_env, _blind_agent_cfg = build_paired_env(
    "TT-Tracking-TerrainBlind-Unitree-G1",
    str(bundle_dir / "pair.json"),
    play=True,
    device="cpu",
    num_envs=1,
    no_terminations=True,
  )
  try:
    oracle_env, _oracle_agent_cfg = build_paired_env(
      task_id,
      str(bundle_dir / "pair.json"),
      play=True,
      device="cpu",
      num_envs=1,
      no_terminations=True,
    )
    try:
      blind_obs, _blind_extras = blind_env.reset()
      oracle_obs, _oracle_extras = oracle_env.reset()

      blind_actor_dim = blind_obs["actor"].shape[-1]
      oracle_actor_dim = oracle_obs["actor"].shape[-1]
      assert oracle_actor_dim >= blind_actor_dim + min_extra_actor_dims

      action_dim = oracle_env.unwrapped.single_action_space.shape[0]
      action = torch.zeros(
        (1, action_dim),
        dtype=torch.float32,
        device=oracle_env.device,
      )
      oracle_obs, reward, terminated, timeouts, extras = oracle_env.step(action)

      assert set(oracle_obs.keys()) == {"actor", "critic"}
      assert oracle_obs["actor"].shape[0] == 1
      assert reward.shape == (1,)
      assert terminated.shape == (1,)
      assert timeouts.shape == (1,)
      assert isinstance(extras, dict)
    finally:
      oracle_env.close()
  finally:
    blind_env.close()
