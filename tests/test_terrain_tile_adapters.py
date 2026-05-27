from __future__ import annotations

import json
from pathlib import Path

from terrain_tracking.runtime.pair_dataset import PairDataset
from terrain_tracking.scene.terrain_adapters.parc import ParcPrimitiveBoxAdapter
from tests.helpers import create_heightfield_collision_manifest, create_motion_clip


def test_parc_adapter_builds_common_terrain_tile(tmp_path: Path) -> None:
  pair_dir = tmp_path / "pair"
  pair_dir.mkdir()
  motion_file = create_motion_clip(pair_dir / "motion.npz", num_frames=4)
  terrain_collision_file = create_heightfield_collision_manifest(
    pair_dir / "terrain_collision.json"
  )
  manifest = tmp_path / "pairs.jsonl"
  manifest.write_text(
    json.dumps(
      {
        "pair_id": "parc/platform/000",
        "source": "parc",
        "motion_file": str(motion_file),
        "terrain_collision_file": str(terrain_collision_file),
        "motion": {
          "fps": 50,
          "frames": 4,
          "root_bounds_xy": [[0.0, -0.2], [0.06, -0.2]],
        },
        "terrain": {
          "adapter": "parc_primitive_boxes",
          "box_count": 4,
          "bounds_xy": [[-0.4, -0.4], [0.4, 0.4]],
        },
      }
    )
    + "\n",
    encoding="utf-8",
  )
  record = PairDataset.load(manifest).records[0]

  tile = ParcPrimitiveBoxAdapter().build_tile(record)

  assert tile.pair_id == record.pair_id
  assert tile.bounds_xy.shape == (2, 2)
  assert tile.terrain_bounds_xy.tolist() == [[-0.4, -0.4], [0.4, 0.4]]
  assert tile.motion_root_bounds_xy.tolist() == [[0.0, -0.2], [0.06, -0.2]]
  assert tile.occupied_bounds_xy.tolist() == [[-0.4, -0.4], [0.4, 0.4]]
  assert len(tile.boxes) == record.terrain.box_count
