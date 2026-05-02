from __future__ import annotations

import json
from pathlib import Path

from terrain_tracking.convert_pair import (
  ConvertPairConfig,
  convert_pair,
)
from terrain_tracking.runtime.pair_manifest import PairManifest
from tests.helpers import create_motion_clip, create_ramp_obj


def test_convert_pair_end_to_end(tmp_path: Path) -> None:
  bundle_dir = convert_pair(
    ConvertPairConfig(
      motion_file=create_motion_clip(tmp_path / "motion.npz"),
      terrain_file=create_ramp_obj(tmp_path / "terrain.obj"),
      output_dir=tmp_path / "converted",
      sample_name="demo_pair",
    )
  )

  assert (bundle_dir / "pair.json").is_file()
  assert (bundle_dir / "meta.json").is_file()

  pair = PairManifest.load(bundle_dir / "pair.json")
  assert pair.motion_file == (tmp_path / "motion.npz").resolve()
  assert pair.terrain_file == (tmp_path / "terrain.obj").resolve()

  payload = json.loads((bundle_dir / "pair.json").read_text(encoding="utf-8"))
  assert payload["motion_file"] == str((tmp_path / "motion.npz").resolve())
  assert payload["terrain_file"] == str((tmp_path / "terrain.obj").resolve())
  assert payload["terrain_translation"] == [0.0, 0.0, 0.0]
  assert payload["terrain_quat_xyzw"] == [0.0, 0.0, 0.0, 1.0]
  assert payload["terrain_scale"] == [1.0, 1.0, 1.0]


def test_convert_pair_preserves_relative_paths_and_terrain_offsets(tmp_path: Path) -> None:
  output_dir = tmp_path / "bundle"
  output_dir.mkdir()
  motion_path = create_motion_clip(output_dir / "motion.npz")
  terrain_path = create_ramp_obj(output_dir / "terrain.obj")

  bundle_dir = convert_pair(
    ConvertPairConfig(
      motion_file=motion_path.name,
      terrain_file=terrain_path.name,
      output_dir=output_dir,
      terrain_translation=(1.0, 2.0, 3.0),
      terrain_quat_xyzw=(0.0, 0.0, 0.70710677, 0.70710677),
      terrain_scale=(0.5, 0.75, 1.25),
    )
  )

  payload = json.loads((bundle_dir / "pair.json").read_text(encoding="utf-8"))
  assert payload["motion_file"] == "motion.npz"
  assert payload["terrain_file"] == "terrain.obj"
  assert payload["terrain_translation"] == [1.0, 2.0, 3.0]
  assert payload["terrain_quat_xyzw"] == [0.0, 0.0, 0.70710677, 0.70710677]
  assert payload["terrain_scale"] == [0.5, 0.75, 1.25]


def test_convert_pair_writes_collision_manifest_reference(tmp_path: Path) -> None:
  motion_path = create_motion_clip(tmp_path / "motion.npz")
  terrain_path = create_ramp_obj(tmp_path / "terrain.obj")
  collision_path = tmp_path / "terrain_collision.json"
  visual_path = create_ramp_obj(tmp_path / "multi_boxes.obj")
  collision_path.write_text("{}", encoding="utf-8")

  bundle_dir = convert_pair(
    ConvertPairConfig(
      motion_file=motion_path,
      terrain_file=terrain_path,
      terrain_collision_file=collision_path,
      terrain_visual_file=visual_path,
      output_dir=tmp_path / "converted",
      sample_name="demo",
    )
  )

  payload = json.loads((bundle_dir / "pair.json").read_text(encoding="utf-8"))
  assert payload["terrain_collision_file"] == str(collision_path.resolve())
  assert payload["terrain_visual_file"] == str(visual_path.resolve())

  pair = PairManifest.load(bundle_dir / "pair.json")
  assert pair.terrain_collision_file == collision_path.resolve()
  assert pair.terrain_visual_file == visual_path.resolve()
