from __future__ import annotations

import json
from pathlib import Path

import pytest

from terrain_tracking.runtime.pair_dataset import PairDataset
from tests.helpers import create_heightfield_collision_manifest, create_motion_clip


def _record(
  tmp_path: Path,
  *,
  pair_id: str,
  frames: int,
  dir_suffix: str = "",
) -> dict[str, object]:
  pair_dir = tmp_path / f"{pair_id.replace('/', '_')}{dir_suffix}"
  pair_dir.mkdir()
  motion_file = create_motion_clip(pair_dir / "motion.npz", num_frames=frames)
  terrain_collision_file = create_heightfield_collision_manifest(
    pair_dir / "terrain_collision.json"
  )
  terrain_visual_file = pair_dir / "multi_boxes.obj"
  terrain_visual_file.write_text("o terrain\n", encoding="utf-8")
  return {
    "pair_id": pair_id,
    "source": "parc",
    "motion_file": str(motion_file),
    "terrain_collision_file": str(terrain_collision_file),
    "terrain_visual_file": str(terrain_visual_file),
    "weight": 1.0,
    "category": "platform",
    "motion": {
      "fps": 50,
      "frames": frames,
      "root_bounds_xy": [[0.0, -0.2], [1.0, 0.2]],
    },
    "terrain": {
      "adapter": "parc_primitive_boxes",
      "box_count": 2,
      "bounds_xy": [[-0.2, -0.2], [0.2, 0.2]],
    },
  }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
  path.write_text(
    "".join(json.dumps(record) + "\n" for record in records),
    encoding="utf-8",
  )
  return path


def test_pair_dataset_loads_jsonl_records(tmp_path: Path) -> None:
  manifest = _write_jsonl(
    tmp_path / "pairs.jsonl",
    [
      _record(tmp_path, pair_id="parc/a/000", frames=4),
      _record(tmp_path, pair_id="parc/b/001", frames=6),
    ],
  )

  dataset = PairDataset.load(manifest, validate="fast")

  assert dataset.pair_ids == ("parc/a/000", "parc/b/001")
  assert dataset.records[0].source == "parc"
  assert dataset.records[0].motion.frames == 4
  assert dataset.records[0].terrain.adapter == "parc_primitive_boxes"


def test_pair_dataset_rejects_duplicate_pair_ids(tmp_path: Path) -> None:
  manifest = _write_jsonl(
    tmp_path / "pairs.jsonl",
    [
      _record(tmp_path, pair_id="parc/a/000", frames=4),
      _record(tmp_path, pair_id="parc/a/000", frames=6, dir_suffix="_dup"),
    ],
  )

  with pytest.raises(ValueError, match="duplicate pair_id"):
    PairDataset.load(manifest, validate="fast")


def test_pair_dataset_rejects_missing_files(tmp_path: Path) -> None:
  record = _record(tmp_path, pair_id="parc/a/000", frames=4)
  record["motion_file"] = str(tmp_path / "missing_motion.npz")
  manifest = _write_jsonl(tmp_path / "pairs.jsonl", [record])

  with pytest.raises(FileNotFoundError, match="missing_motion"):
    PairDataset.load(manifest, validate="fast")
