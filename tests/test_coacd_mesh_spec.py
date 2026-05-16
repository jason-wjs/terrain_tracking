from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from terrain_tracking.runtime.pair_manifest import PairManifest
from terrain_tracking.scene import coacd_mesh_spec
from terrain_tracking.scene.coacd_mesh_spec import (
  CoacdCollisionOptions,
  make_coacd_mesh_spec_fn,
)
from tests.helpers import create_motion_clip, create_quad_obj


def _manifest(tmp_path: Path) -> PairManifest:
  motion_path = create_motion_clip(tmp_path / "motion.npz")
  terrain_path = create_quad_obj(tmp_path / "terrain.obj")
  return PairManifest(
    motion_file=motion_path.resolve(),
    terrain_file=terrain_path.resolve(),
  )


def _tetra_parts() -> list[tuple[np.ndarray, np.ndarray]]:
  return [
    (
      np.array(
        [
          [0.0, 0.0, 0.0],
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 0.4],
        ],
        dtype=np.float32,
      ),
      np.array(
        [
          [0, 1, 2],
          [0, 1, 3],
          [1, 2, 3],
          [2, 0, 3],
        ],
        dtype=np.int32,
      ),
    )
  ]


def test_coacd_spec_adds_visual_mesh_and_collision_hulls(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  manifest = _manifest(tmp_path)
  monkeypatch.setattr(
    coacd_mesh_spec,
    "_load_or_compute_coacd_parts",
    lambda pair, options: _tetra_parts(),
  )
  spec_fn = make_coacd_mesh_spec_fn(
    manifest,
    num_envs=2,
    env_spacing=2.0,
    options=CoacdCollisionOptions(cache_dir=tmp_path / "cache"),
  )

  spec = mujoco.MjSpec()
  spec_fn(spec)

  assert len(spec.meshes) == 2
  paired_bodies = [
    body for body in spec.worldbody.bodies if body.name.startswith("paired_terrain_")
  ]
  assert len(paired_bodies) == 2
  for body in paired_bodies:
    assert len(body.geoms) == 2
    visual, hull = body.geoms
    assert visual.type == mujoco.mjtGeom.mjGEOM_MESH
    assert visual.contype == 0
    assert visual.conaffinity == 0
    assert hull.type == mujoco.mjtGeom.mjGEOM_MESH
    assert hull.contype == 1
    assert hull.conaffinity == 1
    assert hull.margin == 0.0
    assert hull.gap == 0.0


def test_coacd_disk_cache_reuses_saved_parts(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  manifest = _manifest(tmp_path)
  options = CoacdCollisionOptions(cache_dir=tmp_path / "cache")
  calls = 0

  def fake_run(mesh, run_options):
    del mesh, run_options
    nonlocal calls
    calls += 1
    return _tetra_parts()

  monkeypatch.setattr(coacd_mesh_spec, "_run_coacd_decomposition", fake_run)
  first = coacd_mesh_spec._load_or_compute_coacd_parts(manifest, options)
  coacd_mesh_spec._COACD_PARTS_CACHE.clear()
  monkeypatch.setattr(
    coacd_mesh_spec,
    "_run_coacd_decomposition",
    lambda mesh, run_options: pytest.fail("cache miss"),
  )
  second = coacd_mesh_spec._load_or_compute_coacd_parts(manifest, options)

  assert calls == 1
  np.testing.assert_allclose(first[0][0], second[0][0])
  np.testing.assert_array_equal(first[0][1], second[0][1])


def test_missing_coacd_package_reports_backend_name(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  manifest = _manifest(tmp_path)
  mesh = coacd_mesh_spec._load_transformed_trimesh(manifest)
  monkeypatch.setattr(
    coacd_mesh_spec,
    "_import_coacd",
    lambda: (_ for _ in ()).throw(ImportError("missing")),
  )

  with pytest.raises(RuntimeError, match="collision_backend='coacd'.*coacd"):
    coacd_mesh_spec._run_coacd_decomposition(
      mesh,
      CoacdCollisionOptions(cache_dir=tmp_path / "cache"),
    )


def test_sanitize_coacd_parts_rejects_all_degenerate_parts() -> None:
  with pytest.raises(ValueError, match="no valid 3D hulls"):
    coacd_mesh_spec._sanitize_coacd_parts(
      [(np.zeros((3, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.int32))],
      terrain_tag="degenerate",
    )
