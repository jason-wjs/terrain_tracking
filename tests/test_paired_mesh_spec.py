from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from terrain_tracking.runtime.pair_manifest import PairManifest
from terrain_tracking.scene.paired_mesh_spec import (
  compute_env_origins_grid,
  make_paired_mesh_spec_fn,
)
from tests.helpers import create_motion_clip, create_quad_obj


def test_compute_env_origins_grid_matches_mjlab_plane_layout() -> None:
  origins = compute_env_origins_grid(num_envs=4, env_spacing=2.0)
  np.testing.assert_allclose(
    origins,
    np.array(
      [
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [-1.0, 1.0, 0.0],
      ],
      dtype=np.float32,
    ),
  )


def test_make_paired_mesh_spec_fn_applies_transform_and_duplicates_per_env(
  tmp_path: Path,
) -> None:
  motion_path = create_motion_clip(tmp_path / "motion.npz")
  terrain_path = create_quad_obj(tmp_path / "terrain.obj")
  manifest = PairManifest(
    motion_file=motion_path.resolve(),
    terrain_file=terrain_path.resolve(),
    terrain_translation=(0.5, -1.0, 0.2),
    terrain_quat_xyzw=(0.0, 0.0, 0.70710678, 0.70710678),
    terrain_scale=(2.0, 1.0, 1.0),
  )

  spec = mujoco.MjSpec()
  spec_fn = make_paired_mesh_spec_fn(manifest, num_envs=4, env_spacing=2.0)
  spec_fn(spec)

  assert len(spec.meshes) == 1
  assert list(spec.meshes[0].userface) == [0, 1, 2, 0, 2, 3]
  assert list(spec.meshes[0].uservert) == pytest.approx(
    [
      0.5,
      -1.0,
      0.2,
      0.5,
      1.0,
      0.2,
      -0.5,
      1.0,
      0.2,
      -0.5,
      -1.0,
      0.2,
    ]
  )

  assert len(spec.worldbody.bodies) == 4
  expected_origins = compute_env_origins_grid(num_envs=4, env_spacing=2.0)
  for body, expected_origin in zip(spec.worldbody.bodies, expected_origins, strict=True):
    assert len(body.geoms) == 1
    assert body.geoms[0].meshname == spec.meshes[0].name
    np.testing.assert_allclose(np.asarray(body.pos), expected_origin)
