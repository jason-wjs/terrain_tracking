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


def test_train_module_help_mentions_collision_backend() -> None:
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
  assert "--collision-backend" in proc.stdout


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
  assert "--env-spacing" in proc.stdout


def test_play_module_help_mentions_collision_backend() -> None:
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
  assert "--collision-backend" in proc.stdout


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
  assert "--terrain-collision-file" in proc.stdout
  assert "--terrain-visual-file" in proc.stdout
  assert "--output-dir" in proc.stdout
  assert "--terrain-translation" in proc.stdout


def test_convert_omniretarget_module_help_mentions_required_inputs() -> None:
  proc = subprocess.run(
    [
      sys.executable,
      "-m",
      "terrain_tracking.convert_omniretarget_robot_terrain",
      "--help",
    ],
    capture_output=True,
    text=True,
  )
  assert proc.returncode == 0
  assert "--motion-file" in proc.stdout
  assert "--terrain-root" in proc.stdout
  assert "--output-dir" in proc.stdout


def test_convert_pair_shell_uses_collision_manifest_scale_source() -> None:
  script_path = Path(__file__).resolve().parents[1] / "scripts" / "convert_pair.sh"
  script = script_path.read_text(encoding="utf-8")
  assert "--terrain-collision-file" in script
  assert "--terrain-visual-file" in script
  assert "0.7415730337078652" not in script


def test_train_shell_uses_mjlab_env_spacing_flag() -> None:
  script_path = Path(__file__).resolve().parents[1] / "scripts" / "train.sh"
  script = script_path.read_text(encoding="utf-8")
  assert "--env.scene.env-spacing 12.0" in script
  assert "--env-spacing 12.0" not in script
