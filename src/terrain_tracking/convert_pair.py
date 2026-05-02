from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
  path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _validate_manifest_reference(value: str | Path, *, manifest_dir: Path) -> str:
  raw = str(value)
  candidate = Path(raw)
  resolved = (
    candidate.resolve()
    if candidate.is_absolute()
    else (manifest_dir / candidate).resolve()
  )
  if not resolved.exists():
    raise FileNotFoundError(resolved)
  return raw


@dataclass(frozen=True)
class ConvertPairConfig:
  motion_file: str | Path
  terrain_file: str | Path
  output_dir: str | Path
  sample_name: str | None = None
  terrain_collision_file: str | Path | None = None
  terrain_visual_file: str | Path | None = None
  terrain_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
  terrain_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
  terrain_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


def write_pair_manifest_bundle(
  *,
  motion_file: str | Path,
  terrain_file: str | Path,
  output_dir: str | Path,
  sample_name: str | None = None,
  terrain_collision_file: str | Path | None = None,
  terrain_visual_file: str | Path | None = None,
  terrain_translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
  terrain_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
  terrain_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
  source_trace: dict[str, Any] | None = None,
) -> Path:
  output_root = Path(output_dir).resolve()
  bundle_dir = output_root / sample_name if sample_name else output_root
  bundle_dir.mkdir(parents=True, exist_ok=True)

  motion_ref = _validate_manifest_reference(motion_file, manifest_dir=bundle_dir)
  terrain_ref = _validate_manifest_reference(terrain_file, manifest_dir=bundle_dir)
  terrain_collision_ref = (
    _validate_manifest_reference(terrain_collision_file, manifest_dir=bundle_dir)
    if terrain_collision_file is not None
    else None
  )
  terrain_visual_ref = (
    _validate_manifest_reference(terrain_visual_file, manifest_dir=bundle_dir)
    if terrain_visual_file is not None
    else None
  )

  pair_payload: dict[str, Any] = {
    "motion_file": motion_ref,
    "terrain_file": terrain_ref,
    "terrain_translation": [float(v) for v in terrain_translation],
    "terrain_quat_xyzw": [float(v) for v in terrain_quat_xyzw],
    "terrain_scale": [float(v) for v in terrain_scale],
  }
  if terrain_collision_ref is not None:
    pair_payload["terrain_collision_file"] = terrain_collision_ref
  if terrain_visual_ref is not None:
    pair_payload["terrain_visual_file"] = terrain_visual_ref

  _write_json(
    bundle_dir / "pair.json",
    pair_payload,
  )
  _write_json(
    bundle_dir / "meta.json",
    {
      "sample_name": sample_name or bundle_dir.name,
      **(source_trace or {}),
    },
  )
  return bundle_dir.resolve()


def convert_pair(config: ConvertPairConfig) -> Path:
  return write_pair_manifest_bundle(
    motion_file=config.motion_file,
    terrain_file=config.terrain_file,
    output_dir=config.output_dir,
    sample_name=config.sample_name,
    terrain_collision_file=config.terrain_collision_file,
    terrain_visual_file=config.terrain_visual_file,
    terrain_translation=config.terrain_translation,
    terrain_quat_xyzw=config.terrain_quat_xyzw,
    terrain_scale=config.terrain_scale,
    source_trace={
      "motion_file": str(config.motion_file),
      "terrain_obj": str(config.terrain_file),
      "terrain_collision_file": (
        str(config.terrain_collision_file)
        if config.terrain_collision_file is not None
        else None
      ),
      "terrain_visual_file": (
        str(config.terrain_visual_file)
        if config.terrain_visual_file is not None
        else None
      ),
    },
  )


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Compose a terrain_tracking pair manifest from an existing motion file and terrain mesh."
  )
  parser.add_argument("--motion-file", required=True)
  parser.add_argument("--terrain-file", required=True)
  parser.add_argument("--terrain-collision-file", default=None)
  parser.add_argument("--terrain-visual-file", default=None)
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--sample-name", default=None)
  parser.add_argument("--terrain-translation", type=float, nargs=3, default=(0.0, 0.0, 0.0))
  parser.add_argument("--terrain-quat-xyzw", type=float, nargs=4, default=(0.0, 0.0, 0.0, 1.0))
  parser.add_argument("--terrain-scale", type=float, nargs=3, default=(1.0, 1.0, 1.0))
  return parser


def main() -> None:
  args = build_arg_parser().parse_args()
  convert_pair(
    ConvertPairConfig(
      motion_file=args.motion_file,
      terrain_file=args.terrain_file,
      terrain_collision_file=args.terrain_collision_file,
      terrain_visual_file=args.terrain_visual_file,
      output_dir=args.output_dir,
      sample_name=args.sample_name,
      terrain_translation=tuple(args.terrain_translation),
      terrain_quat_xyzw=tuple(args.terrain_quat_xyzw),
      terrain_scale=tuple(args.terrain_scale),
    )
  )


if __name__ == "__main__":
  main()
