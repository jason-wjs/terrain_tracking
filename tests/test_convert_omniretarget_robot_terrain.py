from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from terrain_tracking.convert_omniretarget_robot_terrain import (
  ConvertOmniRetargetRobotTerrainConfig,
  convert_omniretarget_robot_terrain,
  resolve_terrain_urdf,
)
from terrain_tracking.runtime.pair_manifest import PairManifest
from tests.helpers import (
  create_box_obj,
  create_omniretarget_qpos_clip,
  create_omniretarget_terrain_urdf,
)


def test_resolve_terrain_urdf_from_sample_name(tmp_path: Path) -> None:
  terrain_root = tmp_path / "models" / "terrain"
  expected = create_omniretarget_terrain_urdf(
    terrain_root / "climb_00" / "multi_boxes_z_scale_1.0.urdf"
  )

  resolved = resolve_terrain_urdf("climb_00_z_scale_1.0", terrain_root)

  assert resolved == expected.resolve()


def test_convert_omniretarget_robot_terrain_writes_pair_bundle(tmp_path: Path) -> None:
  dataset_root = tmp_path / "OmniRetarget_Dataset"
  motion_path = create_omniretarget_qpos_clip(
    dataset_root / "robot-terrain" / "climb_00_z_scale_1.0.npz",
    num_frames=4,
  )
  terrain_dir = dataset_root / "models" / "terrain" / "climb_00"
  create_box_obj(terrain_dir / "box_models" / "box1.obj")
  terrain_urdf = create_omniretarget_terrain_urdf(
    terrain_dir / "multi_boxes_z_scale_1.0.urdf"
  )

  bundle_dir = convert_omniretarget_robot_terrain(
    ConvertOmniRetargetRobotTerrainConfig(
      motion_file=motion_path,
      terrain_root=dataset_root / "models" / "terrain",
      output_dir=tmp_path / "converted",
      sample_name="climb_00_z_scale_1.0",
    )
  )

  assert bundle_dir == (tmp_path / "converted" / "climb_00_z_scale_1.0").resolve()
  assert (bundle_dir / "motion.npz").is_file()
  assert (bundle_dir / "pair.json").is_file()
  assert (bundle_dir / "meta.json").is_file()

  pair = PairManifest.load(bundle_dir / "pair.json")
  assert pair.motion_file == (bundle_dir / "motion.npz").resolve()
  assert pair.terrain_file == terrain_urdf.resolve()

  motion = np.load(bundle_dir / "motion.npz")
  assert int(np.asarray(motion["fps"]).item()) == 50
  assert motion["joint_pos"].shape == (6, 29)

  meta = json.loads((bundle_dir / "meta.json").read_text(encoding="utf-8"))
  assert meta["sample_name"] == "climb_00_z_scale_1.0"
  assert meta["source_motion_file"] == str(motion_path.resolve())
  assert meta["terrain_file"] == str(terrain_urdf.resolve())
  assert meta["source_fps"] == 30
  assert meta["output_fps"] == 50
  assert meta["source_frame_count"] == 4
  assert meta["output_frame_count"] == 6
