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


def test_convert_shell_dispatches_supported_conversion_kinds() -> None:
  scripts_root = Path(__file__).resolve().parents[1] / "scripts"
  convert_script = (scripts_root / "convert.sh").read_text(encoding="utf-8")
  common = (scripts_root / "lib" / "common.sh").read_text(encoding="utf-8")

  assert "tt_convert_exp" in convert_script
  assert "tt_convert_pair" in common
  assert "tt_convert_omniretarget_robot_terrain" in common
  assert "terrain_tracking.convert_pair" in common
  assert "terrain_tracking.convert_omniretarget_robot_terrain" in common
  assert not (scripts_root / "convert_pair.sh").exists()


def test_train_shell_shared_implementation_uses_mjlab_env_spacing_flag() -> None:
  common_path = Path(__file__).resolve().parents[1] / "scripts" / "lib" / "common.sh"
  common = common_path.read_text(encoding="utf-8")
  train_fn = common.split("tt_train_exp() {", maxsplit=1)[1].split(
    "tt_play_exp() {",
    maxsplit=1,
  )[0]
  assert "--env.scene.env-spacing" in train_fn
  assert "--env-spacing" not in train_fn


def test_exp_scripts_call_shared_launch_functions() -> None:
  scripts_root = Path(__file__).resolve().parents[1] / "scripts"
  for script_path in (scripts_root / "exp").glob("*/*.sh"):
    script = script_path.read_text(encoding="utf-8")
    assert 'source "${SCRIPT_DIR}/../../lib/common.sh"' in script
    assert '"${SCRIPT_DIR}/../../train.sh"' not in script
    assert '"${SCRIPT_DIR}/../../play.sh"' not in script


def test_convert_exp_scripts_call_shared_convert_function() -> None:
  scripts_root = Path(__file__).resolve().parents[1] / "scripts"
  convert_scripts = sorted((scripts_root / "exp" / "convert").glob("*.sh"))
  assert convert_scripts
  assert any("omniretarget" in script.name for script in convert_scripts)
  for script_path in convert_scripts:
    script = script_path.read_text(encoding="utf-8")
    assert "tt_convert_exp" in script
    assert "uv run python -m terrain_tracking.convert_" not in script


def test_obsolete_debug_shell_scripts_are_removed() -> None:
  scripts_root = Path(__file__).resolve().parents[1] / "scripts"
  for name in (
    "debug_play_flat.sh",
    "debug_train_flat.sh",
    "debug_validate_hfield_pair.sh",
    "debug_zero_action_collision_backend.sh",
  ):
    assert not (scripts_root / name).exists()
