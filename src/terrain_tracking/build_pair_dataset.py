from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from terrain_tracking.runtime.pair_dataset import PairDataset
from terrain_tracking.runtime.terrain_collision import TerrainCollisionManifest
from terrain_tracking.scene.primitive_box_terrain import build_primitive_box_tile


@dataclass(frozen=True)
class BuildPairDatasetConfig:
  root: Path
  output: Path
  source: Literal["parc"] = "parc"
  include_path_parts: tuple[str, ...] = ()
  pair_dir_name_prefix: str | None = None


def build_parc_pair_dataset(config: BuildPairDatasetConfig) -> PairDataset:
  root = config.root.expanduser().resolve()
  if not root.exists():
    raise FileNotFoundError(root)

  collision_index = _terrain_collision_index(root)
  records = []
  for motion_file in sorted(root.rglob("motion.npz")):
    if not _matches_filters(
      root,
      motion_file.parent,
      include_path_parts=config.include_path_parts,
      pair_dir_name_prefix=config.pair_dir_name_prefix,
    ):
      continue
    terrain_collision_file = _find_terrain_collision_file(
      root,
      motion_file.parent,
      collision_index,
    )
    if terrain_collision_file is None:
      continue
    records.append(
      _build_parc_record(
        root,
        motion_file,
        terrain_collision_file=terrain_collision_file,
        source=config.source,
      )
    )
  if not records:
    raise ValueError(f"no PARC motion/terrain pairs found under {root}")

  output = config.output.expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(
    "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
    encoding="utf-8",
  )
  return PairDataset.load(output, validate="fast")


def _build_parc_record(
  root: Path,
  motion_file: Path,
  *,
  terrain_collision_file: Path,
  source: str,
) -> dict[str, object]:
  visual_file = terrain_collision_file.parent / "multi_boxes.obj"
  dataset_relative_dir = _dataset_relative_pair_dir(root, motion_file.parent)
  pair_id = f"{source}/{dataset_relative_dir.as_posix()}"

  motion = _motion_diagnostics(motion_file)
  terrain = _terrain_diagnostics(terrain_collision_file)
  record: dict[str, object] = {
    "pair_id": pair_id,
    "source": source,
    "motion_file": str(motion_file.resolve()),
    "terrain_collision_file": str(terrain_collision_file.resolve()),
    "weight": 1.0,
    "category": _category_from_relative_dir(dataset_relative_dir),
    "motion": motion,
    "terrain": terrain,
  }
  if visual_file.exists():
    record["terrain_visual_file"] = str(visual_file.resolve())
  return record


def _dataset_relative_pair_dir(root: Path, pair_dir: Path) -> Path:
  relative = pair_dir.relative_to(root)
  if "paired" not in relative.parts:
    return relative
  paired_idx = relative.parts.index("paired")
  return Path(*relative.parts[paired_idx + 1 :])


def _terrain_collision_index(root: Path) -> dict[str, list[Path]]:
  index: dict[str, list[Path]] = {}
  for collision_file in sorted(root.rglob("terrain_collision.json")):
    index.setdefault(collision_file.parent.name, []).append(collision_file)
  return index


def _find_terrain_collision_file(
  root: Path,
  motion_dir: Path,
  collision_index: dict[str, list[Path]],
) -> Path | None:
  same_dir_collision = motion_dir / "terrain_collision.json"
  if same_dir_collision.exists():
    return same_dir_collision

  candidates = collision_index.get(motion_dir.name, [])
  if not candidates:
    return None
  if len(candidates) == 1:
    return candidates[0]

  motion_parts = set(motion_dir.relative_to(root).parts)
  return max(
    candidates,
    key=lambda candidate: (
      len(motion_parts.intersection(candidate.parent.relative_to(root).parts)),
      -len(candidate.parent.relative_to(root).parts),
      str(candidate),
    ),
  )


def _matches_filters(
  root: Path,
  pair_dir: Path,
  *,
  include_path_parts: tuple[str, ...],
  pair_dir_name_prefix: str | None,
) -> bool:
  relative_posix = pair_dir.relative_to(root).as_posix()
  if include_path_parts and not all(part in relative_posix for part in include_path_parts):
    return False
  if pair_dir_name_prefix is not None and not pair_dir.name.startswith(
    pair_dir_name_prefix
  ):
    return False
  return True


def _category_from_relative_dir(relative: Path) -> str | None:
  if not relative.parts:
    return None
  if relative.parts[0] == "mj" and len(relative.parts) >= 2:
    return relative.parts[1]
  return relative.parts[0]


def _motion_diagnostics(motion_file: Path) -> dict[str, object]:
  with np.load(motion_file) as data:
    if "body_pos_w" not in data:
      raise ValueError(f"{motion_file} missing body_pos_w")
    body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float64)
    if body_pos_w.ndim != 3 or body_pos_w.shape[0] == 0 or body_pos_w.shape[2] < 2:
      raise ValueError(f"{motion_file} body_pos_w must have shape (frames, bodies, 3)")
    root_xy = body_pos_w[:, 0, :2]
    fps = float(np.asarray(data["fps"]).item()) if "fps" in data else 50.0
  bounds = np.stack([np.min(root_xy, axis=0), np.max(root_xy, axis=0)], axis=0)
  return {
    "fps": fps,
    "frames": int(body_pos_w.shape[0]),
    "root_bounds_xy": _json_bounds(bounds),
  }


def _terrain_diagnostics(terrain_collision_file: Path) -> dict[str, object]:
  manifest = TerrainCollisionManifest.load(terrain_collision_file)
  tile = build_primitive_box_tile(manifest)
  mins = []
  maxs = []
  for box in tile.boxes:
    pos = np.asarray(box.pos[:2], dtype=np.float64)
    size = np.asarray(box.size[:2], dtype=np.float64)
    mins.append(pos - size)
    maxs.append(pos + size)
  bounds = np.stack(
    [np.min(np.stack(mins, axis=0), axis=0), np.max(np.stack(maxs, axis=0), axis=0)],
    axis=0,
  )
  return {
    "adapter": "parc_primitive_boxes",
    "box_count": tile.diagnostics.box_count,
    "bounds_xy": _json_bounds(bounds),
  }


def _json_bounds(bounds: np.ndarray) -> list[list[float]]:
  return [[float(value) for value in row] for row in bounds]


def entry_point() -> None:
  parser = argparse.ArgumentParser(description="Build a JSONL pair dataset manifest.")
  parser.add_argument("--source", choices=("parc",), default="parc")
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--include-path-part",
    action="append",
    default=[],
    help="Only include pair directories whose root-relative path contains this string.",
  )
  parser.add_argument(
    "--pair-dir-name-prefix",
    default=None,
    help="Only include pair directories whose basename starts with this prefix.",
  )
  args = parser.parse_args()

  build_parc_pair_dataset(
    BuildPairDatasetConfig(root=args.root, output=args.output, source=args.source)
    if not args.include_path_part and args.pair_dir_name_prefix is None
    else BuildPairDatasetConfig(
      root=args.root,
      output=args.output,
      source=args.source,
      include_path_parts=tuple(args.include_path_part),
      pair_dir_name_prefix=args.pair_dir_name_prefix,
    )
  )


if __name__ == "__main__":
  entry_point()


__all__ = [
  "BuildPairDatasetConfig",
  "build_parc_pair_dataset",
  "entry_point",
]
