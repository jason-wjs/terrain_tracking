from __future__ import annotations

import json
from pathlib import Path

from terrain_tracking.convert_pair import write_pair_manifest_bundle
from terrain_tracking.runtime.pair_manifest import PairManifest
from tests.helpers import create_motion_clip, create_quad_obj


def test_write_pair_bundle_creates_runtime_bundle(tmp_path: Path) -> None:
  motion_path = create_motion_clip(tmp_path / "source_motion.npz")
  terrain_path = create_quad_obj(tmp_path / "source_terrain.obj")

  bundle_dir = write_pair_manifest_bundle(
    motion_file=motion_path,
    terrain_file=terrain_path,
    output_dir=tmp_path / "output",
    sample_name="platform_001",
    source_trace={
      "motion_file": str(motion_path),
      "terrain_obj": str(terrain_path),
    },
  )

  assert bundle_dir == (tmp_path / "output" / "platform_001").resolve()

  payload = json.loads((bundle_dir / "pair.json").read_text(encoding="utf-8"))
  assert payload["motion_file"] == str(motion_path.resolve())
  assert payload["terrain_file"] == str(terrain_path.resolve())
  assert payload["terrain_translation"] == [0.0, 0.0, 0.0]
  assert payload["terrain_quat_xyzw"] == [0.0, 0.0, 0.0, 1.0]
  assert payload["terrain_scale"] == [1.0, 1.0, 1.0]

  meta = json.loads((bundle_dir / "meta.json").read_text(encoding="utf-8"))
  assert meta["sample_name"] == "platform_001"
  assert meta["motion_file"] == str(motion_path)
  assert meta["terrain_obj"] == str(terrain_path)

  manifest = PairManifest.load(bundle_dir / "pair.json")
  assert manifest.motion_file == motion_path.resolve()
  assert manifest.terrain_file == terrain_path.resolve()
