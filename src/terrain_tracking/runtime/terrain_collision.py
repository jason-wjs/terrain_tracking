from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


def _required_float_pair(payload: dict[str, Any], key: str) -> tuple[float, float]:
  value = payload.get(key)
  if not isinstance(value, (list, tuple)) or len(value) != 2:
    raise ValueError(f"collision.{key} must contain 2 numbers")
  return (float(value[0]), float(value[1]))


@dataclass(frozen=True)
class TerrainCollisionManifest:
  type: Literal["heightfield"]
  hf_file: Path
  min_point: tuple[float, float]
  dx: float
  base_z: float
  xy_scale: float = 1.0
  height_scale: float = 1.0

  @classmethod
  def load(cls, path: str | Path) -> "TerrainCollisionManifest":
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
      raise ValueError("Terrain collision manifest must be a JSON object")
    collision = payload.get("collision")
    if not isinstance(collision, dict):
      raise ValueError("Terrain collision manifest is missing collision object")
    if collision.get("type") != "heightfield":
      raise ValueError(f"Unsupported terrain collision type: {collision.get('type')!r}")

    hf_value = collision.get("hf_file")
    if not isinstance(hf_value, str) or not hf_value:
      raise ValueError("collision.hf_file must be a non-empty string")
    hf_file = (manifest_path.parent / hf_value).resolve()
    if not hf_file.exists():
      raise FileNotFoundError(hf_file)

    return cls(
      type="heightfield",
      hf_file=hf_file,
      min_point=_required_float_pair(collision, "min_point"),
      dx=float(collision["dx"]),
      base_z=float(collision["base_z"]),
      xy_scale=float(collision.get("xy_scale", 1.0)),
      height_scale=float(collision.get("height_scale", 1.0)),
    )
