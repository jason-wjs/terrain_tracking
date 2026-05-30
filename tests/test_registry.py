from __future__ import annotations

import importlib
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import mjlab.tasks.registry as registry
import pytest
import tomllib


@pytest.fixture(autouse=True)
def preserve_registry() -> Iterator[None]:
  original_registry = dict(registry._REGISTRY)
  try:
    yield
  finally:
    registry._REGISTRY.clear()
    registry._REGISTRY.update(original_registry)


def test_pyproject_declares_mjlab_task_entrypoint() -> None:
  pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
  value = pyproject["project"]["entry-points"]["mjlab.tasks"]["terrain_tracking"]
  assert value == "terrain_tracking._mjlab_tasks"


def test_root_package_import_has_no_registration_side_effect() -> None:
  code = (
    "import mjlab.tasks.registry as registry; "
    "registry._REGISTRY.clear(); "
    "import terrain_tracking; "
    "print(registry.list_tasks())"
  )
  proc = subprocess.run(
    [sys.executable, "-c", code],
    check=True,
    capture_output=True,
    text=True,
  )
  assert proc.stdout.strip() == "[]"


def test_blind_terrain_tracking_tasks_register_with_mjlab_registry() -> None:
  registry._REGISTRY.clear()
  from terrain_tracking.tasks.blind_terrain_tracking.config import g1 as g1_config

  importlib.reload(g1_config)

  tasks = registry.list_tasks()
  assert "TT-Tracking-TerrainBlind-Unitree-G1" in tasks
  assert "TT-Tracking-TerrainBlind-Unitree-G1-No-State-Estimation" in tasks


def test_oracle_terrain_tracking_tasks_register_with_mjlab_registry() -> None:
  registry._REGISTRY.clear()
  from terrain_tracking.tasks.oracle_terrain_tracking.config import g1 as g1_config

  importlib.reload(g1_config)

  tasks = registry.list_tasks()
  assert "TT-Tracking-TerrainOracleHeight-Unitree-G1" in tasks
  assert "TT-Tracking-TerrainOracleHeightLongScanPhpReward-Unitree-G1" in tasks
  assert "TT-Tracking-TerrainOracleTeacher-Unitree-G1" in tasks
  assert not any("TerrainOracle" in task and "No-State-Estimation" in task for task in tasks)


def test_general_oracle_teacher_task_registers_with_mjlab_registry() -> None:
  registry._REGISTRY.clear()
  from terrain_tracking.tasks.general_terrain_tracking.config import g1 as g1_config

  importlib.reload(g1_config)

  tasks = registry.list_tasks()
  assert "TT-Tracking-TerrainOracleTeacherGeneral-Unitree-G1" in tasks
  assert "TT-Tracking-TerrainOracleTeacherGeneral-Unitree-G1-Play" in tasks
  assert "TT-Tracking-TerrainOracleHeightLongScanPhpRewardGeneral-Unitree-G1" in tasks
  assert (
    "TT-Tracking-TerrainOracleHeightLongScanPhpRewardGeneral-Unitree-G1-Play"
    in tasks
  )
