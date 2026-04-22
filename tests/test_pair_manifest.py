from __future__ import annotations

from pathlib import Path

import pytest

from terrain_tracking.runtime.pair_manifest import PairManifest
from tests.helpers import create_motion_clip, create_pair_manifest, create_quad_obj


def test_pair_manifest_load_resolves_relative_paths(tmp_path: Path) -> None:
  assets_dir = tmp_path / "assets"
  assets_dir.mkdir()
  motion_path = create_motion_clip(assets_dir / "motion.npz")
  terrain_path = create_quad_obj(assets_dir / "terrain.obj")

  manifest_dir = tmp_path / "pairs"
  manifest_dir.mkdir()
  manifest_path = create_pair_manifest(
    manifest_dir / "pair.json",
    motion_file="../assets/motion.npz",
    terrain_file="../assets/terrain.obj",
  )

  manifest = PairManifest.load(manifest_path)

  assert manifest.motion_file == motion_path.resolve()
  assert manifest.terrain_file == terrain_path.resolve()
  assert manifest.terrain_translation == (0.0, 0.0, 0.0)
  assert manifest.terrain_quat_xyzw == (0.0, 0.0, 0.0, 1.0)
  assert manifest.terrain_scale == (1.0, 1.0, 1.0)


def test_pair_manifest_load_reads_optional_transform_fields(tmp_path: Path) -> None:
  motion_path = create_motion_clip(tmp_path / "motion.npz")
  terrain_path = create_quad_obj(tmp_path / "terrain.obj")
  manifest_path = create_pair_manifest(
    tmp_path / "pair.json",
    motion_file=motion_path.name,
    terrain_file=terrain_path.name,
    terrain_translation=(0.5, -1.0, 0.25),
    terrain_quat_xyzw=(0.0, 0.0, 0.70710678, 0.70710678),
    terrain_scale=(2.0, 1.5, 1.0),
  )

  manifest = PairManifest.load(manifest_path)

  assert manifest.terrain_translation == (0.5, -1.0, 0.25)
  assert manifest.terrain_quat_xyzw == pytest.approx(
    (0.0, 0.0, 0.70710678, 0.70710678)
  )
  assert manifest.terrain_scale == (2.0, 1.5, 1.0)


def test_pair_manifest_load_rejects_missing_required_keys(tmp_path: Path) -> None:
  create_motion_clip(tmp_path / "motion.npz")

  manifest_path = tmp_path / "pair.json"
  manifest_path.write_text('{"motion_file": "motion.npz"}', encoding="utf-8")

  with pytest.raises(ValueError, match="terrain_file"):
    PairManifest.load(manifest_path)


def test_pair_manifest_load_rejects_missing_target_files(tmp_path: Path) -> None:
  manifest_path = create_pair_manifest(
    tmp_path / "pair.json",
    motion_file="missing_motion.npz",
    terrain_file="missing_terrain.obj",
  )

  with pytest.raises(FileNotFoundError, match="missing_motion.npz"):
    PairManifest.load(manifest_path)
