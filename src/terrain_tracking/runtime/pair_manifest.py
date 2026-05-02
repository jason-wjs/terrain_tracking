from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _required_path(
  payload: dict[str, Any], *, key: str, manifest_path: Path
) -> Path:
  if key not in payload:
    raise ValueError(f"Pair manifest is missing required key: {key}")
  value = payload[key]
  if not isinstance(value, str) or not value:
    raise ValueError(f"Pair manifest key '{key}' must be a non-empty string")
  path = (manifest_path.parent / value).resolve()
  if not path.exists():
    raise FileNotFoundError(path)
  return path


def _optional_path(
  payload: dict[str, Any], *, key: str, manifest_path: Path
) -> Path | None:
  value = payload.get(key)
  if value is None:
    return None
  if not isinstance(value, str) or not value:
    raise ValueError(f"Pair manifest key '{key}' must be a non-empty string")
  path = (manifest_path.parent / value).resolve()
  if not path.exists():
    raise FileNotFoundError(path)
  return path


def _float_tuple(
  payload: dict[str, Any],
  *,
  key: str,
  length: int,
  default: tuple[float, ...],
) -> tuple[float, ...]:
  value = payload.get(key, default)
  if not isinstance(value, (list, tuple)) or len(value) != length:
    raise ValueError(f"Pair manifest key '{key}' must contain {length} numbers")
  return tuple(float(v) for v in value)


@dataclass(frozen=True)
class PairManifest:
  motion_file: Path
  terrain_file: Path
  terrain_collision_file: Path | None = None
  terrain_visual_file: Path | None = None
  terrain_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
  terrain_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
  terrain_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

  @classmethod
  def load(cls, manifest_path: str | Path) -> "PairManifest":
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
      raise ValueError("Pair manifest must be a JSON object")
    return cls(
      motion_file=_required_path(
        payload,
        key="motion_file",
        manifest_path=manifest_path,
      ),
      terrain_file=_required_path(
        payload,
        key="terrain_file",
        manifest_path=manifest_path,
      ),
      terrain_collision_file=_optional_path(
        payload,
        key="terrain_collision_file",
        manifest_path=manifest_path,
      ),
      terrain_visual_file=_optional_path(
        payload,
        key="terrain_visual_file",
        manifest_path=manifest_path,
      ),
      terrain_translation=_float_tuple(
        payload,
        key="terrain_translation",
        length=3,
        default=(0.0, 0.0, 0.0),
      ),
      terrain_quat_xyzw=_float_tuple(
        payload,
        key="terrain_quat_xyzw",
        length=4,
        default=(0.0, 0.0, 0.0, 1.0),
      ),
      terrain_scale=_float_tuple(
        payload,
        key="terrain_scale",
        length=3,
        default=(1.0, 1.0, 1.0),
      ),
    )
