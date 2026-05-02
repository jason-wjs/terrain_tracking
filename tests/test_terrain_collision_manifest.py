from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from terrain_tracking.runtime.terrain_collision import TerrainCollisionManifest


def test_load_heightfield_collision_manifest_resolves_relative_hf(tmp_path: Path) -> None:
  np.save(
    tmp_path / "terrain_hf.npy",
    np.array([[0.0, 0.2], [0.4, 0.6]], dtype=np.float32),
  )
  manifest_path = tmp_path / "terrain_collision.json"
  manifest_path.write_text(
    json.dumps(
      {
        "schema_version": 1,
        "terrain_name": "demo",
        "frame": {"convention": "z_up", "origin": "motion_world"},
        "collision": {
          "type": "heightfield",
          "hf_file": "terrain_hf.npy",
          "min_point": [-1.0, -2.0],
          "dx": 0.4,
          "base_z": -0.4,
          "xy_scale": 0.5,
          "height_scale": 0.75,
        },
      }
    ),
    encoding="utf-8",
  )

  manifest = TerrainCollisionManifest.load(manifest_path)

  assert manifest.type == "heightfield"
  assert manifest.hf_file == (tmp_path / "terrain_hf.npy").resolve()
  assert manifest.min_point == (-1.0, -2.0)
  assert manifest.dx == 0.4
  assert manifest.base_z == -0.4
  assert manifest.xy_scale == 0.5
  assert manifest.height_scale == 0.75
