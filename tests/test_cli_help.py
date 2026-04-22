from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_train_module_help_mentions_pair_manifest() -> None:
  proc = subprocess.run(
    [
      sys.executable,
      "-m",
      "terrain_tracking.tasks.blind_terrain_tracking.scripts.train",
      "--help",
    ],
    check=True,
    capture_output=True,
    text=True,
  )
  assert "--pair-manifest" in proc.stdout


def test_play_module_help_mentions_pair_manifest() -> None:
  proc = subprocess.run(
    [
      sys.executable,
      "-m",
      "terrain_tracking.tasks.blind_terrain_tracking.scripts.play",
      "--help",
    ],
    check=True,
    capture_output=True,
    text=True,
  )
  assert "--pair-manifest" in proc.stdout


def test_convert_module_help_mentions_required_inputs() -> None:
  proc = subprocess.run(
    [
      sys.executable,
      "-m",
      "terrain_tracking.convert_pair",
      "--help",
    ],
    capture_output=True,
    text=True,
  )
  assert proc.returncode == 0
  assert "--motion-file" in proc.stdout
  assert "--terrain-file" in proc.stdout
  assert "--output-dir" in proc.stdout
  assert "--terrain-translation" in proc.stdout


def test_convert_pair_shell_uses_calibrated_terrain_scale() -> None:
  script_path = Path(__file__).resolve().parents[1] / "scripts" / "convert_pair.sh"
  script = script_path.read_text(encoding="utf-8")
  assert "--terrain-scale 1 1 1" not in script
  assert script.count("--terrain-scale 0.7415730337078652 0.7415730337078652 0.7415730337078652") == 2
