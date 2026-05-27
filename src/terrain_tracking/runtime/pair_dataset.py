from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

BoundsXY = tuple[tuple[float, float], tuple[float, float]]
ValidationMode = Literal["fast", "strict"]


@dataclass(frozen=True)
class MotionDiagnostics:
  fps: float
  frames: int
  root_bounds_xy: BoundsXY


@dataclass(frozen=True)
class TerrainDiagnostics:
  adapter: str
  box_count: int
  bounds_xy: BoundsXY


@dataclass(frozen=True)
class PairDatasetRecord:
  pair_id: str
  source: str
  motion_file: Path
  terrain_collision_file: Path
  terrain_visual_file: Path | None
  weight: float
  category: str | None
  motion: MotionDiagnostics
  terrain: TerrainDiagnostics


@dataclass(frozen=True)
class PairDataset:
  path: Path
  records: tuple[PairDatasetRecord, ...]

  @property
  def pair_ids(self) -> tuple[str, ...]:
    return tuple(record.pair_id for record in self.records)

  @classmethod
  def load(
    cls,
    path: str | Path,
    *,
    validate: ValidationMode = "fast",
  ) -> "PairDataset":
    if validate not in ("fast", "strict"):
      raise ValueError(f"unsupported validation mode: {validate!r}")

    manifest_path = Path(path).expanduser().resolve()
    records = tuple(_load_jsonl(manifest_path))
    _validate_unique_pair_ids(records)
    return cls(path=manifest_path, records=records)


def _load_jsonl(path: Path) -> list[PairDatasetRecord]:
  records: list[PairDatasetRecord] = []
  for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    stripped = line.strip()
    if not stripped:
      continue
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
      raise ValueError(f"line {line_no}: record must be a JSON object")
    records.append(_record_from_payload(payload, manifest_dir=path.parent, line_no=line_no))
  if not records:
    raise ValueError(f"pair dataset is empty: {path}")
  return records


def _record_from_payload(
  payload: dict[str, Any],
  *,
  manifest_dir: Path,
  line_no: int,
) -> PairDatasetRecord:
  pair_id = _required_str(payload, "pair_id", line_no=line_no)
  source = _required_str(payload, "source", line_no=line_no)
  motion_file = _required_existing_path(
    payload,
    "motion_file",
    manifest_dir=manifest_dir,
    line_no=line_no,
  )
  terrain_collision_file = _required_existing_path(
    payload,
    "terrain_collision_file",
    manifest_dir=manifest_dir,
    line_no=line_no,
  )
  terrain_visual_file = _optional_existing_path(
    payload,
    "terrain_visual_file",
    manifest_dir=manifest_dir,
    line_no=line_no,
  )
  return PairDatasetRecord(
    pair_id=pair_id,
    source=source,
    motion_file=motion_file,
    terrain_collision_file=terrain_collision_file,
    terrain_visual_file=terrain_visual_file,
    weight=float(payload.get("weight", 1.0)),
    category=_optional_str(payload, "category", line_no=line_no),
    motion=_motion_diagnostics(payload, line_no=line_no),
    terrain=_terrain_diagnostics(payload, line_no=line_no),
  )


def _motion_diagnostics(payload: dict[str, Any], *, line_no: int) -> MotionDiagnostics:
  motion = _required_object(payload, "motion", line_no=line_no)
  frames = int(_required_number(motion, "frames", line_no=line_no))
  if frames <= 0:
    raise ValueError(f"line {line_no}: motion.frames must be positive")
  return MotionDiagnostics(
    fps=float(_required_number(motion, "fps", line_no=line_no)),
    frames=frames,
    root_bounds_xy=_required_bounds(motion, "root_bounds_xy", line_no=line_no),
  )


def _terrain_diagnostics(
  payload: dict[str, Any],
  *,
  line_no: int,
) -> TerrainDiagnostics:
  terrain = _required_object(payload, "terrain", line_no=line_no)
  box_count = int(_required_number(terrain, "box_count", line_no=line_no))
  if box_count < 0:
    raise ValueError(f"line {line_no}: terrain.box_count must be non-negative")
  return TerrainDiagnostics(
    adapter=_required_str(terrain, "adapter", line_no=line_no),
    box_count=box_count,
    bounds_xy=_required_bounds(terrain, "bounds_xy", line_no=line_no),
  )


def _validate_unique_pair_ids(records: tuple[PairDatasetRecord, ...]) -> None:
  seen: set[str] = set()
  for record in records:
    if record.pair_id in seen:
      raise ValueError(f"duplicate pair_id: {record.pair_id}")
    seen.add(record.pair_id)


def _required_object(
  payload: dict[str, Any],
  key: str,
  *,
  line_no: int,
) -> dict[str, Any]:
  value = payload.get(key)
  if not isinstance(value, dict):
    raise ValueError(f"line {line_no}: {key} must be an object")
  return value


def _required_str(payload: dict[str, Any], key: str, *, line_no: int) -> str:
  value = payload.get(key)
  if not isinstance(value, str) or not value:
    raise ValueError(f"line {line_no}: {key} must be a non-empty string")
  return value


def _optional_str(payload: dict[str, Any], key: str, *, line_no: int) -> str | None:
  value = payload.get(key)
  if value is None:
    return None
  if not isinstance(value, str) or not value:
    raise ValueError(f"line {line_no}: {key} must be a non-empty string")
  return value


def _required_number(payload: dict[str, Any], key: str, *, line_no: int) -> float:
  value = payload.get(key)
  if not isinstance(value, int | float):
    raise ValueError(f"line {line_no}: {key} must be a number")
  return float(value)


def _required_bounds(
  payload: dict[str, Any],
  key: str,
  *,
  line_no: int,
) -> BoundsXY:
  value = payload.get(key)
  if (
    not isinstance(value, list | tuple)
    or len(value) != 2
    or not all(isinstance(row, list | tuple) and len(row) == 2 for row in value)
  ):
    raise ValueError(f"line {line_no}: {key} must be [[min_x, min_y], [max_x, max_y]]")
  bounds = (
    (float(value[0][0]), float(value[0][1])),
    (float(value[1][0]), float(value[1][1])),
  )
  if bounds[1][0] < bounds[0][0] or bounds[1][1] < bounds[0][1]:
    raise ValueError(f"line {line_no}: {key} max corner must be >= min corner")
  return bounds


def _required_existing_path(
  payload: dict[str, Any],
  key: str,
  *,
  manifest_dir: Path,
  line_no: int,
) -> Path:
  value = _required_str(payload, key, line_no=line_no)
  path = _resolve_path(value, manifest_dir)
  if not path.exists():
    raise FileNotFoundError(path)
  return path


def _optional_existing_path(
  payload: dict[str, Any],
  key: str,
  *,
  manifest_dir: Path,
  line_no: int,
) -> Path | None:
  value = payload.get(key)
  if value in (None, ""):
    return None
  if not isinstance(value, str):
    raise ValueError(f"line {line_no}: {key} must be a string")
  path = _resolve_path(value, manifest_dir)
  if not path.exists():
    raise FileNotFoundError(path)
  return path


def _resolve_path(value: str, manifest_dir: Path) -> Path:
  path = Path(value).expanduser()
  if not path.is_absolute():
    path = manifest_dir / path
  return path.resolve()


__all__ = [
  "BoundsXY",
  "MotionDiagnostics",
  "PairDataset",
  "PairDatasetRecord",
  "TerrainDiagnostics",
  "ValidationMode",
]
