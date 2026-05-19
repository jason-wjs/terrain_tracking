from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from terrain_tracking.convert_pair import write_pair_manifest_bundle
from terrain_tracking.omniretarget_motion import (
  emit_mjlab_motion_npz,
  load_omniretarget_clip,
  resample_clip,
)


@dataclass(frozen=True)
class ConvertOmniRetargetRobotTerrainConfig:
  motion_file: str | Path
  terrain_root: str | Path
  output_dir: str | Path
  sample_name: str | None = None
  output_fps: int = 50


def resolve_terrain_urdf(sample_name: str, terrain_root: str | Path) -> Path:
  marker = "_z_scale_"
  if marker not in sample_name:
    raise ValueError(
      "OmniRetarget robot-terrain sample name must contain '_z_scale_': "
      f"{sample_name!r}"
    )
  terrain_name = sample_name.split(marker, maxsplit=1)[0]
  z_scale = sample_name[len(terrain_name) :]
  terrain_file = (
    Path(terrain_root) / terrain_name / f"multi_boxes{z_scale}.urdf"
  ).resolve()
  if not terrain_file.exists():
    raise FileNotFoundError(terrain_file)
  return terrain_file


def _write_json(path: Path, payload: dict[str, object]) -> None:
  path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def convert_omniretarget_robot_terrain(
  config: ConvertOmniRetargetRobotTerrainConfig,
) -> Path:
  motion_file = Path(config.motion_file).resolve()
  sample_name = config.sample_name or motion_file.stem
  terrain_file = resolve_terrain_urdf(sample_name, config.terrain_root)

  source_clip = load_omniretarget_clip(motion_file)
  output_clip = resample_clip(source_clip, output_fps=config.output_fps)

  bundle_dir = Path(config.output_dir).resolve() / sample_name
  bundle_dir.mkdir(parents=True, exist_ok=True)
  output_motion_file = bundle_dir / "motion.npz"
  emit_mjlab_motion_npz(
    motion_file,
    output_motion_file,
    output_fps=config.output_fps,
  )

  bundle_dir = write_pair_manifest_bundle(
    motion_file="motion.npz",
    terrain_file=terrain_file,
    output_dir=config.output_dir,
    sample_name=sample_name,
    source_trace={
      "source_motion_file": str(motion_file),
      "terrain_file": str(terrain_file),
    },
  )

  motion = np.load(output_motion_file)
  meta_path = bundle_dir / "meta.json"
  meta = json.loads(meta_path.read_text(encoding="utf-8"))
  meta.update(
    {
      "source_motion_file": str(motion_file),
      "terrain_file": str(terrain_file),
      "source_fps": int(source_clip.fps),
      "output_fps": int(np.asarray(motion["fps"]).item()),
      "source_frame_count": int(source_clip.qpos.shape[0]),
      "output_frame_count": int(output_clip.qpos.shape[0]),
    }
  )
  _write_json(meta_path, meta)
  return bundle_dir


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Convert one OmniRetarget robot-terrain sample into a terrain_tracking pair bundle."
  )
  parser.add_argument("--motion-file", required=True)
  parser.add_argument("--terrain-root", required=True)
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--sample-name", default=None)
  parser.add_argument("--output-fps", type=int, default=50)
  return parser


def main() -> None:
  args = build_arg_parser().parse_args()
  convert_omniretarget_robot_terrain(
    ConvertOmniRetargetRobotTerrainConfig(
      motion_file=args.motion_file,
      terrain_root=args.terrain_root,
      output_dir=args.output_dir,
      sample_name=args.sample_name,
      output_fps=args.output_fps,
    )
  )


if __name__ == "__main__":
  main()
