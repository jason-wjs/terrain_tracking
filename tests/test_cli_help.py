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


def test_general_train_module_help_mentions_pair_dataset() -> None:
  proc = subprocess.run(
    [
      sys.executable,
      "-m",
      "terrain_tracking.tasks.general_terrain_tracking.scripts.train",
      "--help",
    ],
    check=True,
    capture_output=True,
    text=True,
  )
  assert "--pair-dataset" in proc.stdout
  assert "--pair-sampler-mode" in proc.stdout


def test_general_play_module_help_mentions_pair_dataset() -> None:
  proc = subprocess.run(
    [
      sys.executable,
      "-m",
      "terrain_tracking.tasks.general_terrain_tracking.scripts.play",
      "--help",
    ],
    check=True,
    capture_output=True,
    text=True,
  )
  assert "--pair-dataset" in proc.stdout
  assert "--dataset-validate" in proc.stdout


def test_general_frontends_copy_current_sampler_cfg_fields() -> None:
  scripts_root = Path(__file__).resolve().parents[1] / "src" / "terrain_tracking"
  train_script = (
    scripts_root / "tasks" / "general_terrain_tracking" / "scripts" / "train.py"
  ).read_text(encoding="utf-8")
  play_script = (
    scripts_root / "tasks" / "general_terrain_tracking" / "scripts" / "play.py"
  ).read_text(encoding="utf-8")

  for script in (train_script, play_script):
    assert "adaptive_alpha=motion_cmd.sampler.adaptive_alpha" in script
    assert (
      "adaptive_uniform_ratio=motion_cmd.sampler.adaptive_uniform_ratio" in script
    )
    assert "ema_alpha" not in script


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


def test_general_mid_blocks_exp_scripts_target_pair_dataset_training() -> None:
  scripts_root = Path(__file__).resolve().parents[1] / "scripts"
  common = (scripts_root / "lib" / "common.sh").read_text(encoding="utf-8")
  convert_script = (
    scripts_root / "exp" / "convert" / "parc_mid_blocks_dataset.sh"
  ).read_text(encoding="utf-8")
  train_script = (
    scripts_root / "exp" / "train" / "mid_blocks_general_oracle_teacher_adaptive.sh"
  ).read_text(encoding="utf-8")
  height_longscan_train_script = (
    scripts_root
    / "exp"
    / "train"
    / "mid_blocks_general_oracle_height_longscan_phpreward_adaptive.sh"
  ).read_text(encoding="utf-8")

  assert "tt_convert_parc_pair_dataset" in common
  assert "terrain_tracking.build_pair_dataset" in common
  assert "tt_train_general_pair_dataset_exp" in common
  assert "terrain_tracking.tasks.general_terrain_tracking.scripts.train" in common
  assert "--pair-dir-name-prefix" in common
  assert "--include-path-part" in common
  assert "CONVERT_KIND=\"${CONVERT_KIND:-parc_pair_dataset}\"" in convert_script
  assert "PAIR_DIR_NAME_PREFIX=\"${PAIR_DIR_NAME_PREFIX:-mid_blocks}\"" in convert_script
  assert "INCLUDE_PATH_PARTS=\"${INCLUDE_PATH_PARTS:-mj/mid_climbing}\"" in convert_script
  assert "tt_convert_exp" in convert_script
  assert "TT-Tracking-TerrainOracleTeacherGeneral-Unitree-G1" in train_script
  assert "PAIR_DATASET=\"${PAIR_DATASET:-" in train_script
  assert "tt_train_general_pair_dataset_exp" in train_script
  assert (
    "TT-Tracking-TerrainOracleHeightLongScanPhpRewardGeneral-Unitree-G1"
    in height_longscan_train_script
  )
  assert "pair_dataset_mid_blocks.jsonl" in height_longscan_train_script
  assert (
    "mid_blocks_general_g1_oracle_height_longscan_phpreward_n16384_adaptive"
    in height_longscan_train_script
  )
  assert (
    "PAIR_SAMPLER_MODE=\"${PAIR_SAMPLER_MODE:-independent}\""
    in height_longscan_train_script
  )
  assert "tt_train_general_pair_dataset_exp" in height_longscan_train_script


def test_obsolete_debug_shell_scripts_are_removed() -> None:
  scripts_root = Path(__file__).resolve().parents[1] / "scripts"
  for name in (
    "debug_play_flat.sh",
    "debug_train_flat.sh",
    "debug_validate_hfield_pair.sh",
    "debug_zero_action_collision_backend.sh",
  ):
    assert not (scripts_root / name).exists()
