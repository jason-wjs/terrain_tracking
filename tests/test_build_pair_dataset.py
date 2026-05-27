from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from terrain_tracking.build_pair_dataset import (
  BuildPairDatasetConfig,
  build_parc_pair_dataset,
)
from tests.helpers import create_heightfield_collision_manifest, create_motion_clip


def _create_parc_pair(root: Path, relative: str, *, frames: int) -> Path:
  pair_dir = root / relative
  pair_dir.mkdir(parents=True)
  create_motion_clip(pair_dir / "motion.npz", num_frames=frames)
  create_heightfield_collision_manifest(pair_dir / "terrain_collision.json")
  (pair_dir / "multi_boxes.obj").write_text("o terrain\n", encoding="utf-8")
  return pair_dir


def _read_jsonl(path: Path) -> list[dict[str, object]]:
  return [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
  ]


def test_build_parc_pair_dataset_writes_manifest_with_diagnostics(
  tmp_path: Path,
) -> None:
  root = tmp_path / "parc_initial_aug_g1"
  _create_parc_pair(root, "platform/sample_000", frames=4)
  _create_parc_pair(root, "stairs/sample_001", frames=6)
  output = tmp_path / "pair_dataset.jsonl"

  dataset = build_parc_pair_dataset(
    BuildPairDatasetConfig(root=root, output=output, source="parc")
  )

  assert dataset.pair_ids == ("parc/platform/sample_000", "parc/stairs/sample_001")
  records = _read_jsonl(output)
  assert records[0]["pair_id"] == "parc/platform/sample_000"
  assert records[0]["category"] == "platform"
  assert records[0]["motion"]["frames"] == 4
  np.testing.assert_allclose(
    records[0]["motion"]["root_bounds_xy"],
    [[0.0, -0.2], [0.06, -0.2]],
  )
  assert records[0]["terrain"]["adapter"] == "parc_primitive_boxes"
  assert records[0]["terrain"]["box_count"] == 4
  assert records[0]["terrain"]["bounds_xy"] == [[-0.4, -0.4], [0.4, 0.4]]


def test_build_parc_pair_dataset_skips_unmatched_terrain_files(tmp_path: Path) -> None:
  root = tmp_path / "parc_initial_aug_g1"
  _create_parc_pair(root, "platform/sample_000", frames=4)
  unmatched = root / "platform" / "terrain_only"
  unmatched.mkdir(parents=True)
  create_heightfield_collision_manifest(unmatched / "terrain_collision.json")
  output = tmp_path / "pair_dataset.jsonl"

  dataset = build_parc_pair_dataset(
    BuildPairDatasetConfig(root=root, output=output, source="parc")
  )

  assert dataset.pair_ids == ("parc/platform/sample_000",)


def test_build_parc_pair_dataset_filters_by_path_part_and_pair_dir_prefix(
  tmp_path: Path,
) -> None:
  root = tmp_path / "parc_initial_aug_g1"
  _create_parc_pair(root, "mj/mid_climbing/mid_blocks_001", frames=4)
  _create_parc_pair(root, "mj/mid_climbing/flipped/mid_blocks_001_flipped", frames=5)
  _create_parc_pair(root, "mj/mid_climbing/other_blocks_001", frames=6)
  _create_parc_pair(root, "mj/platform/mid_blocks_wrong_category", frames=7)
  output = tmp_path / "pair_dataset_mid_blocks.jsonl"

  dataset = build_parc_pair_dataset(
    BuildPairDatasetConfig(
      root=root,
      output=output,
      source="parc",
      include_path_parts=("mj/mid_climbing",),
      pair_dir_name_prefix="mid_blocks",
    )
  )

  assert dataset.pair_ids == (
    "parc/mj/mid_climbing/flipped/mid_blocks_001_flipped",
    "parc/mj/mid_climbing/mid_blocks_001",
  )
  assert dataset.records[0].category == "mid_climbing"


def test_build_parc_pair_dataset_matches_split_parc_mj_and_workspace_layout(
  tmp_path: Path,
) -> None:
  root = tmp_path / "parc_initial_aug_g1"
  motion_dir = root / "mj" / "mid_climbing" / "flipped" / "mid_blocks_001_flipped"
  motion_dir.mkdir(parents=True)
  create_motion_clip(motion_dir / "motion.npz", num_frames=5)
  terrain_dir = (
    root
    / "parc_process"
    / "workspace"
    / "mid_climbing"
    / "flipped"
    / "workspace"
    / "mid_blocks_001_flipped"
  )
  terrain_dir.mkdir(parents=True)
  create_heightfield_collision_manifest(terrain_dir / "terrain_collision.json")
  (terrain_dir / "multi_boxes.obj").write_text("o terrain\n", encoding="utf-8")
  output = tmp_path / "pair_dataset_mid_blocks.jsonl"

  dataset = build_parc_pair_dataset(
    BuildPairDatasetConfig(
      root=root,
      output=output,
      source="parc",
      include_path_parts=("mj/mid_climbing",),
      pair_dir_name_prefix="mid_blocks",
    )
  )

  assert dataset.pair_ids == ("parc/mj/mid_climbing/flipped/mid_blocks_001_flipped",)
  assert dataset.records[0].terrain_collision_file == (
    terrain_dir / "terrain_collision.json"
  ).resolve()
  assert dataset.records[0].terrain_visual_file == (terrain_dir / "multi_boxes.obj").resolve()
