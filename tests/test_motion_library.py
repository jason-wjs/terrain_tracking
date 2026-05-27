from __future__ import annotations

import json
from pathlib import Path

import torch

from terrain_tracking.runtime.motion_library import MotionLibrary
from terrain_tracking.runtime.pair_dataset import PairDataset
from tests.helpers import create_heightfield_collision_manifest, create_motion_clip


def _record(tmp_path: Path, pair_id: str, frames: int) -> dict[str, object]:
  pair_dir = tmp_path / pair_id.replace("/", "_")
  pair_dir.mkdir()
  motion_file = create_motion_clip(pair_dir / "motion.npz", num_frames=frames)
  terrain_collision_file = create_heightfield_collision_manifest(
    pair_dir / "terrain_collision.json"
  )
  return {
    "pair_id": pair_id,
    "source": "parc",
    "motion_file": str(motion_file),
    "terrain_collision_file": str(terrain_collision_file),
    "motion": {
      "fps": 50,
      "frames": frames,
      "root_bounds_xy": [[0.0, -0.2], [0.1, -0.2]],
    },
    "terrain": {
      "adapter": "parc_primitive_boxes",
      "box_count": 4,
      "bounds_xy": [[-0.4, -0.4], [0.4, 0.4]],
    },
  }


def test_motion_library_uses_concat_ragged_frame_layout(tmp_path: Path) -> None:
  manifest = tmp_path / "pairs.jsonl"
  manifest.write_text(
    "".join(
      json.dumps(record) + "\n"
      for record in [
        _record(tmp_path, "parc/a/000", 4),
        _record(tmp_path, "parc/b/001", 7),
      ]
    ),
    encoding="utf-8",
  )
  records = PairDataset.load(manifest).records

  library = MotionLibrary.from_records(records, device="cpu")
  global_idx = library.global_frame_index(
    pair_indices=torch.tensor([1]),
    local_frames=torch.tensor([2]),
  )

  assert int(global_idx[0]) == int(library.frame_offsets[1]) + 2
  assert library.pair_ids == ("parc/a/000", "parc/b/001")
  assert library.frame_counts.tolist() == [4, 7]
  assert library.joint_pos(global_idx).shape == (1, 29)
