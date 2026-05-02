from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from terrain_tracking.runtime.terrain_collision import TerrainCollisionManifest
from terrain_tracking.scene.heightfield_spec import make_heightfield_spec_fn


def _write_collision(tmp_path: Path) -> TerrainCollisionManifest:
  np.save(
    tmp_path / "terrain_hf.npy",
    np.array([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32),
  )
  path = tmp_path / "terrain_collision.json"
  path.write_text(
    json.dumps(
      {
        "schema_version": 1,
        "terrain_name": "demo",
        "collision": {
          "type": "heightfield",
          "hf_file": "terrain_hf.npy",
          "min_point": [-0.2, -0.2],
          "dx": 0.4,
          "base_z": -0.4,
          "xy_scale": 1.0,
          "height_scale": 1.0,
        },
      }
    ),
    encoding="utf-8",
  )
  return TerrainCollisionManifest.load(path)


def test_make_heightfield_spec_fn_adds_hfield_geom_per_env(tmp_path: Path) -> None:
  manifest = _write_collision(tmp_path)
  spec = mujoco.MjSpec()
  spec_fn = make_heightfield_spec_fn(manifest, num_envs=3, env_spacing=2.0)

  spec_fn(spec)

  assert len(spec.hfields) == 1
  assert spec.nconmax == 384
  assert spec.njmax == 384
  assert spec.hfields[0].nrow == 9
  assert spec.hfields[0].ncol == 9
  assert spec.hfields[0].size[0] == 0.4
  assert spec.hfields[0].size[1] == 0.4
  paired_bodies = [
    body for body in spec.worldbody.bodies if body.name.startswith("paired_terrain_")
  ]
  assert len(paired_bodies) == 3
  assert all(body.geoms[0].type == mujoco.mjtGeom.mjGEOM_HFIELD for body in paired_bodies)
  assert paired_bodies[0].pos[0] == 2.0
  assert paired_bodies[0].pos[1] == 0.0


def test_heightfield_spec_compiles_to_hfield_model(tmp_path: Path) -> None:
  manifest = _write_collision(tmp_path)
  spec = mujoco.MjSpec()
  make_heightfield_spec_fn(manifest, num_envs=1, env_spacing=2.0)(spec)

  model = spec.compile()

  assert model.nhfield == 1
  assert model.ngeom == 1
  assert model.geom_type[0] == mujoco.mjtGeom.mjGEOM_HFIELD


def test_heightfield_spec_maps_parc_x_to_mujoco_columns(tmp_path: Path) -> None:
  np.save(
    tmp_path / "terrain_hf.npy",
    np.zeros((2, 3), dtype=np.float32),
  )
  path = tmp_path / "terrain_collision.json"
  path.write_text(
    json.dumps(
      {
        "schema_version": 1,
        "terrain_name": "demo",
        "collision": {
          "type": "heightfield",
          "hf_file": "terrain_hf.npy",
          "min_point": [0.0, 0.0],
          "dx": 0.5,
          "base_z": -0.5,
          "xy_scale": 1.0,
          "height_scale": 1.0,
        },
      }
    ),
    encoding="utf-8",
  )
  manifest = TerrainCollisionManifest.load(path)
  spec = mujoco.MjSpec()

  make_heightfield_spec_fn(manifest, num_envs=1, env_spacing=2.0)(spec)

  assert spec.hfields[0].nrow == 13
  assert spec.hfields[0].ncol == 9
  assert spec.hfields[0].size[0] == 0.5
  assert spec.hfields[0].size[1] == 0.75
