from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from terrain_tracking.runtime.terrain_collision import TerrainCollisionManifest
from terrain_tracking.scene.primitive_box_terrain import (
  build_primitive_box_tile,
  make_primitive_box_tile_spec_fn,
)


def _write_collision(tmp_path: Path, hf: np.ndarray) -> TerrainCollisionManifest:
  np.save(tmp_path / "terrain_hf.npy", hf.astype(np.float32))
  manifest_path = tmp_path / "terrain_collision.json"
  manifest_path.write_text(
    json.dumps(
      {
        "schema_version": 1,
        "terrain_name": "demo",
        "collision": {
          "type": "heightfield",
          "hf_file": "terrain_hf.npy",
          "min_point": [-0.8, -0.8],
          "dx": 0.4,
          "base_z": -0.4,
          "xy_scale": 0.5,
          "height_scale": 0.5,
        },
      }
    ),
    encoding="utf-8",
  )
  return TerrainCollisionManifest.load(manifest_path)


def test_build_primitive_box_tile_merges_constant_height_rectangles(
  tmp_path: Path,
) -> None:
  hf = np.array(
    [
      [0.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 1.0, 0.0],
      [0.0, 1.0, 1.0, 0.0],
    ],
    dtype=np.float32,
  )
  manifest = _write_collision(tmp_path, hf)

  tile = build_primitive_box_tile(manifest)

  assert tile.diagnostics.hf_shape == (3, 4)
  assert tile.diagnostics.unique_height_count == 2
  assert tile.diagnostics.nonzero_cell_count == 4
  assert tile.diagnostics.box_count == 2
  assert tile.boxes[0].name == "terrain_base"
  assert tile.boxes[1].name == "terrain_h0_rect0"
  assert tile.boxes[0].size == pytest.approx((0.3, 0.4, 0.2))
  assert tile.boxes[0].pos == pytest.approx((-0.2, -0.1, -0.2))
  assert tile.boxes[1].size == pytest.approx((0.2, 0.2, 0.25))
  assert tile.boxes[1].pos == pytest.approx((-0.1, -0.1, 0.25))
  assert tile.footprint == pytest.approx((0.6, 0.8))


def test_make_primitive_box_tile_spec_fn_compiles_one_tile_not_per_env(
  tmp_path: Path,
) -> None:
  hf = np.array(
    [
      [0.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 1.0, 0.0],
      [0.0, 1.0, 1.0, 0.0],
    ],
    dtype=np.float32,
  )
  manifest = _write_collision(tmp_path, hf)
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="terrain")

  make_primitive_box_tile_spec_fn(manifest)(spec, body)
  model = spec.compile()

  assert model.ngeom == 2
  assert model.geom_type[0] == mujoco.mjtGeom.mjGEOM_BOX
  assert model.geom_type[1] == mujoco.mjtGeom.mjGEOM_BOX
