# Oracle Terrain Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two oracle perception terrain-tracking task variants for Unitree G1: one with a clean local height scan and one with the height scan plus PHP teacher-style global tracking privileged state.

**Architecture:** Add a new `oracle_terrain_tracking` task family that reuses the existing blind terrain tracking environment and PPO config, then layers oracle observations on top. Keep blind tracking behavior unchanged, register the new tasks through the existing `terrain_tracking._mjlab_tasks` entry point, and reuse the existing train/play scripts by passing the new task IDs.

**Tech Stack:** Python, mjlab task registry, mjlab `RayCastSensorCfg`, mjlab observation manager, MuJoCo/MuJoCo-Warp raycasting, pytest, uv.

---

## File Structure

- Create `src/terrain_tracking/tasks/oracle_terrain_tracking/__init__.py`
  - Package marker and short docstring for the oracle task family.
- Create `src/terrain_tracking/tasks/oracle_terrain_tracking/config/__init__.py`
  - Package marker for oracle task configs.
- Create `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/__init__.py`
  - Registers the two oracle Unitree G1 task IDs with mjlab.
- Create `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/env_cfgs.py`
  - Reuses blind env cfg and adds oracle observations.
- Create `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/rl_cfg.py`
  - Reuses blind PPO config and changes experiment metadata.
- Create `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/observations.py`
  - Defines teacher privileged observation functions.
- Modify `src/terrain_tracking/_mjlab_tasks.py`
  - Imports the oracle G1 config module so entry-point loading registers the new tasks.
- Modify `tests/test_registry.py`
  - Adds registry coverage for oracle task IDs.
- Create `tests/test_oracle_terrain_tracking_config.py`
  - Covers height scan config, teacher privileged terms, blind isolation, and teacher observation functions.
- Modify `tests/test_env_smoke.py`
  - Adds reset/step smoke coverage for both oracle task IDs.
- Modify `README.md`
  - Documents how to train the oracle tasks with the existing train script.

---

### Task 1: Register Oracle Task Skeleton

**Files:**
- Modify: `tests/test_registry.py`
- Create: `src/terrain_tracking/tasks/oracle_terrain_tracking/__init__.py`
- Create: `src/terrain_tracking/tasks/oracle_terrain_tracking/config/__init__.py`
- Create: `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/__init__.py`
- Create: `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/env_cfgs.py`
- Create: `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/rl_cfg.py`
- Modify: `src/terrain_tracking/_mjlab_tasks.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing registry test**

Append this test to `tests/test_registry.py`:

```python
def test_oracle_terrain_tracking_tasks_register_with_mjlab_registry() -> None:
  registry._REGISTRY.clear()
  from terrain_tracking.tasks.oracle_terrain_tracking.config import g1 as g1_config

  importlib.reload(g1_config)

  tasks = registry.list_tasks()
  assert "TT-Tracking-TerrainOracleHeight-Unitree-G1" in tasks
  assert "TT-Tracking-TerrainOracleTeacher-Unitree-G1" in tasks
  assert not any("TerrainOracle" in task and "No-State-Estimation" in task for task in tasks)
```

- [ ] **Step 2: Run registry tests to verify failure**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_registry.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'terrain_tracking.tasks.oracle_terrain_tracking'`.

- [ ] **Step 3: Create oracle package skeleton**

Create `src/terrain_tracking/tasks/oracle_terrain_tracking/__init__.py`:

```python
"""Oracle terrain tracking task family."""
```

Create `src/terrain_tracking/tasks/oracle_terrain_tracking/config/__init__.py`:

```python
"""Task configuration package for oracle terrain tracking."""
```

Create `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/rl_cfg.py`:

```python
from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from terrain_tracking.tasks.blind_terrain_tracking.config.g1.rl_cfg import (
  unitree_g1_blind_terrain_tracking_ppo_runner_cfg,
)


def unitree_g1_oracle_terrain_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_g1_blind_terrain_tracking_ppo_runner_cfg()
  cfg.experiment_name = "g1_oracle_terrain_tracking"
  return cfg
```

Create `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/env_cfgs.py`:

```python
from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from terrain_tracking.tasks.blind_terrain_tracking.config.g1.env_cfgs import (
  unitree_g1_blind_terrain_tracking_env_cfg,
)


def unitree_g1_oracle_height_terrain_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return unitree_g1_blind_terrain_tracking_env_cfg(play=play)


def unitree_g1_oracle_teacher_terrain_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return unitree_g1_oracle_height_terrain_tracking_env_cfg(play=play)
```

Create `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/__init__.py`:

```python
"""Task registration for Unitree G1 oracle terrain tracking."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  unitree_g1_oracle_height_terrain_tracking_env_cfg,
  unitree_g1_oracle_teacher_terrain_tracking_env_cfg,
)
from .rl_cfg import unitree_g1_oracle_terrain_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="TT-Tracking-TerrainOracleHeight-Unitree-G1",
  env_cfg=unitree_g1_oracle_height_terrain_tracking_env_cfg(),
  play_env_cfg=unitree_g1_oracle_height_terrain_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_oracle_terrain_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="TT-Tracking-TerrainOracleTeacher-Unitree-G1",
  env_cfg=unitree_g1_oracle_teacher_terrain_tracking_env_cfg(),
  play_env_cfg=unitree_g1_oracle_teacher_terrain_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_oracle_terrain_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
```

- [ ] **Step 4: Import oracle config from entry-point bootstrap**

Replace `src/terrain_tracking/_mjlab_tasks.py` with:

```python
"""Bootstrap module loaded through the ``mjlab.tasks`` entry-point group."""

from terrain_tracking.tasks.blind_terrain_tracking.config.g1 import *  # noqa: F401,F403
from terrain_tracking.tasks.oracle_terrain_tracking.config.g1 import *  # noqa: F401,F403
```

- [ ] **Step 5: Run registry tests to verify pass**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_registry.py
```

Expected: PASS.

- [ ] **Step 6: Commit task skeleton**

Run:

```bash
git add tests/test_registry.py src/terrain_tracking/_mjlab_tasks.py src/terrain_tracking/tasks/oracle_terrain_tracking
git commit -m "feat: register oracle terrain tracking tasks"
```

---

### Task 2: Add Oracle Height Scan Config

**Files:**
- Create: `tests/test_oracle_terrain_tracking_config.py`
- Modify: `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/env_cfgs.py`
- Test: `tests/test_oracle_terrain_tracking_config.py`

- [ ] **Step 1: Write failing config tests for oracle height scan**

Create `tests/test_oracle_terrain_tracking_config.py`:

```python
from __future__ import annotations

import torch
from mjlab.envs import mdp as envs_mdp
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg

from terrain_tracking.tasks.blind_terrain_tracking.config.g1.env_cfgs import (
  unitree_g1_blind_terrain_tracking_env_cfg,
)
from terrain_tracking.tasks.oracle_terrain_tracking.config.g1.env_cfgs import (
  unitree_g1_oracle_height_terrain_tracking_env_cfg,
  unitree_g1_oracle_teacher_terrain_tracking_env_cfg,
)

ORACLE_TEACHER_TERMS = {
  "global_anchor_pos_error_w",
  "global_anchor_lin_vel_error_w",
  "reference_anchor_lin_vel_w",
}


def _term_names(cfg, group_name: str) -> set[str]:
  return set(cfg.observations[group_name].terms)


def test_blind_config_does_not_include_oracle_observations() -> None:
  cfg = unitree_g1_blind_terrain_tracking_env_cfg()

  for group_name in ("actor", "critic"):
    term_names = _term_names(cfg, group_name)
    assert "height_scan" not in term_names
    assert ORACLE_TEACHER_TERMS.isdisjoint(term_names)

  sensor_names = {sensor.name for sensor in cfg.scene.sensors or ()}
  assert "terrain_scan" not in sensor_names


def test_oracle_height_adds_clean_height_scan_to_actor_and_critic() -> None:
  cfg = unitree_g1_oracle_height_terrain_tracking_env_cfg()

  for group_name in ("actor", "critic"):
    term = cfg.observations[group_name].terms["height_scan"]
    assert term.func is envs_mdp.height_scan
    assert term.params == {"sensor_name": "terrain_scan"}
    assert term.noise is None
    assert term.scale == 0.2
    assert term.delay_min_lag == 0
    assert term.delay_max_lag == 0

  sensor_by_name = {sensor.name: sensor for sensor in cfg.scene.sensors or ()}
  terrain_scan = sensor_by_name["terrain_scan"]
  assert isinstance(terrain_scan, RayCastSensorCfg)
  assert isinstance(terrain_scan.frame, ObjRef)
  assert terrain_scan.frame.type == "body"
  assert terrain_scan.frame.name == "torso_link"
  assert terrain_scan.frame.entity == "robot"
  assert terrain_scan.ray_alignment == "yaw"
  assert terrain_scan.max_distance == 5.0
  assert terrain_scan.include_geom_groups == (0,)
  assert isinstance(terrain_scan.pattern, GridPatternCfg)
  assert terrain_scan.pattern.size == (0.7, 0.7)
  assert terrain_scan.pattern.resolution == 0.1


def test_oracle_height_grid_pattern_has_expected_current_ray_count() -> None:
  cfg = unitree_g1_oracle_height_terrain_tracking_env_cfg()
  sensor_by_name = {sensor.name: sensor for sensor in cfg.scene.sensors or ()}
  terrain_scan = sensor_by_name["terrain_scan"]
  assert isinstance(terrain_scan, RayCastSensorCfg)
  assert isinstance(terrain_scan.pattern, GridPatternCfg)

  offsets, directions = terrain_scan.pattern.generate_rays(None, "cpu")

  assert offsets.shape == (64, 3)
  assert directions.shape == (64, 3)
  assert torch.allclose(directions, torch.tensor([[0.0, 0.0, -1.0]]).repeat(64, 1))


def test_oracle_teacher_includes_height_scan_before_teacher_terms_exist() -> None:
  cfg = unitree_g1_oracle_teacher_terrain_tracking_env_cfg()

  for group_name in ("actor", "critic"):
    assert "height_scan" in _term_names(cfg, group_name)
```

- [ ] **Step 2: Run config tests to verify failure**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_oracle_terrain_tracking_config.py
```

Expected: FAIL because `height_scan` is not in the oracle observations yet.

- [ ] **Step 3: Add height scan sensor and observation terms**

Replace `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/env_cfgs.py` with:

```python
from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg

from terrain_tracking.tasks.blind_terrain_tracking.config.g1.env_cfgs import (
  unitree_g1_blind_terrain_tracking_env_cfg,
)

TERRAIN_SCAN_SENSOR_NAME = "terrain_scan"
HEIGHT_SCAN_MAX_DISTANCE = 5.0


def _height_scan_term() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=envs_mdp.height_scan,
    params={"sensor_name": TERRAIN_SCAN_SENSOR_NAME},
    scale=1.0 / HEIGHT_SCAN_MAX_DISTANCE,
  )


def _add_oracle_height_scan(cfg: ManagerBasedRlEnvCfg) -> None:
  terrain_scan = RayCastSensorCfg(
    name=TERRAIN_SCAN_SENSOR_NAME,
    frame=ObjRef(type="body", name="torso_link", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(0.7, 0.7), resolution=0.1),
    max_distance=HEIGHT_SCAN_MAX_DISTANCE,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (terrain_scan,)

  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms["height_scan"] = _height_scan_term()


def unitree_g1_oracle_height_terrain_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_blind_terrain_tracking_env_cfg(play=play)
  _add_oracle_height_scan(cfg)
  return cfg


def unitree_g1_oracle_teacher_terrain_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return unitree_g1_oracle_height_terrain_tracking_env_cfg(play=play)
```

- [ ] **Step 4: Run config tests to verify pass**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_oracle_terrain_tracking_config.py
```

Expected: PASS.

- [ ] **Step 5: Run registry tests to catch registration regressions**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_registry.py
```

Expected: PASS.

- [ ] **Step 6: Commit height scan config**

Run:

```bash
git add tests/test_oracle_terrain_tracking_config.py src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/env_cfgs.py
git commit -m "feat: add oracle terrain height scan config"
```

---

### Task 3: Add Oracle Teacher Privileged Terms

**Files:**
- Create: `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/observations.py`
- Modify: `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/env_cfgs.py`
- Modify: `tests/test_oracle_terrain_tracking_config.py`
- Test: `tests/test_oracle_terrain_tracking_config.py`

- [ ] **Step 1: Add failing tests for teacher terms and functions**

Append these imports to `tests/test_oracle_terrain_tracking_config.py`:

```python
from terrain_tracking.tasks.oracle_terrain_tracking.config.g1 import (
  observations as oracle_obs,
)
```

Append these helper classes and tests to `tests/test_oracle_terrain_tracking_config.py`:

```python
class _FakeMotionCommand:
  def __init__(self) -> None:
    self.anchor_pos_w = torch.tensor(
      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
      dtype=torch.float32,
    )
    self.robot_anchor_pos_w = torch.tensor(
      [[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]],
      dtype=torch.float32,
    )
    self.anchor_lin_vel_w = torch.tensor(
      [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
      dtype=torch.float32,
    )
    self.robot_anchor_lin_vel_w = torch.tensor(
      [[0.0, 0.1, 0.1], [0.1, 0.2, 0.3]],
      dtype=torch.float32,
    )


class _FakeCommandManager:
  def __init__(self, command: _FakeMotionCommand) -> None:
    self.command = command

  def get_term(self, command_name: str) -> _FakeMotionCommand:
    assert command_name == "motion"
    return self.command


class _FakeEnv:
  def __init__(self, command: _FakeMotionCommand) -> None:
    self.command_manager = _FakeCommandManager(command)


def test_teacher_privileged_observation_functions_return_expected_values() -> None:
  command = _FakeMotionCommand()
  env = _FakeEnv(command)

  assert torch.allclose(
    oracle_obs.global_anchor_pos_error_w(env, "motion"),
    torch.full((2, 3), 0.5),
  )
  assert torch.allclose(
    oracle_obs.global_anchor_lin_vel_error_w(env, "motion"),
    torch.tensor([[0.1, 0.1, 0.2], [0.3, 0.3, 0.3]], dtype=torch.float32),
  )
  assert torch.allclose(
    oracle_obs.reference_anchor_lin_vel_w(env, "motion"),
    command.anchor_lin_vel_w,
  )


def test_oracle_height_does_not_include_teacher_privileged_terms() -> None:
  cfg = unitree_g1_oracle_height_terrain_tracking_env_cfg()

  for group_name in ("actor", "critic"):
    assert ORACLE_TEACHER_TERMS.isdisjoint(_term_names(cfg, group_name))


def test_oracle_teacher_adds_privileged_terms_to_actor_and_critic() -> None:
  cfg = unitree_g1_oracle_teacher_terrain_tracking_env_cfg()

  expected_funcs = {
    "global_anchor_pos_error_w": oracle_obs.global_anchor_pos_error_w,
    "global_anchor_lin_vel_error_w": oracle_obs.global_anchor_lin_vel_error_w,
    "reference_anchor_lin_vel_w": oracle_obs.reference_anchor_lin_vel_w,
  }
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    assert "height_scan" in terms
    for term_name, func in expected_funcs.items():
      term = terms[term_name]
      assert term.func is func
      assert term.params == {"command_name": "motion"}
      assert term.noise is None
```

- [ ] **Step 2: Run config tests to verify failure**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_oracle_terrain_tracking_config.py
```

Expected: FAIL with `ImportError` or missing teacher privileged terms.

- [ ] **Step 3: Implement teacher privileged observation functions**

Create `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/observations.py`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.tasks.tracking.mdp import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _motion_command(env: ManagerBasedRlEnv, command_name: str) -> MotionCommand:
  return cast(MotionCommand, env.command_manager.get_term(command_name))


def global_anchor_pos_error_w(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = _motion_command(env, command_name)
  return command.anchor_pos_w - command.robot_anchor_pos_w


def global_anchor_lin_vel_error_w(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = _motion_command(env, command_name)
  return command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w


def reference_anchor_lin_vel_w(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = _motion_command(env, command_name)
  return command.anchor_lin_vel_w
```

- [ ] **Step 4: Add teacher terms to oracle teacher env config**

Replace `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/env_cfgs.py` with:

```python
from __future__ import annotations

from collections.abc import Callable

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg

from terrain_tracking.tasks.blind_terrain_tracking.config.g1.env_cfgs import (
  unitree_g1_blind_terrain_tracking_env_cfg,
)
from terrain_tracking.tasks.oracle_terrain_tracking.config.g1 import observations

TERRAIN_SCAN_SENSOR_NAME = "terrain_scan"
HEIGHT_SCAN_MAX_DISTANCE = 5.0
TEACHER_COMMAND_NAME = "motion"


def _height_scan_term() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=envs_mdp.height_scan,
    params={"sensor_name": TERRAIN_SCAN_SENSOR_NAME},
    scale=1.0 / HEIGHT_SCAN_MAX_DISTANCE,
  )


def _teacher_term(func: Callable[[ManagerBasedRlEnv, str], torch.Tensor]) -> ObservationTermCfg:
  return ObservationTermCfg(
    func=func,
    params={"command_name": TEACHER_COMMAND_NAME},
  )


def _add_oracle_height_scan(cfg: ManagerBasedRlEnvCfg) -> None:
  terrain_scan = RayCastSensorCfg(
    name=TERRAIN_SCAN_SENSOR_NAME,
    frame=ObjRef(type="body", name="torso_link", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(0.7, 0.7), resolution=0.1),
    max_distance=HEIGHT_SCAN_MAX_DISTANCE,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (terrain_scan,)

  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms["height_scan"] = _height_scan_term()


def _add_oracle_teacher_terms(cfg: ManagerBasedRlEnvCfg) -> None:
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    terms["global_anchor_pos_error_w"] = _teacher_term(
      observations.global_anchor_pos_error_w
    )
    terms["global_anchor_lin_vel_error_w"] = _teacher_term(
      observations.global_anchor_lin_vel_error_w
    )
    terms["reference_anchor_lin_vel_w"] = _teacher_term(
      observations.reference_anchor_lin_vel_w
    )


def unitree_g1_oracle_height_terrain_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_blind_terrain_tracking_env_cfg(play=play)
  _add_oracle_height_scan(cfg)
  return cfg


def unitree_g1_oracle_teacher_terrain_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_oracle_height_terrain_tracking_env_cfg(play=play)
  _add_oracle_teacher_terms(cfg)
  return cfg
```

- [ ] **Step 5: Run config tests to verify pass**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_oracle_terrain_tracking_config.py
```

Expected: PASS.

- [ ] **Step 6: Run registry tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_registry.py
```

Expected: PASS.

- [ ] **Step 7: Commit teacher privileged observations**

Run:

```bash
git add tests/test_oracle_terrain_tracking_config.py src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1
git commit -m "feat: add oracle teacher privileged observations"
```

---

### Task 4: Add Oracle Environment Smoke Tests

**Files:**
- Modify: `tests/test_env_smoke.py`
- Test: `tests/test_env_smoke.py`

- [ ] **Step 1: Add failing env smoke test for oracle tasks**

Append this test to `tests/test_env_smoke.py`:

```python
@pytest.mark.parametrize(
  ("task_id", "min_extra_actor_dims"),
  [
    ("TT-Tracking-TerrainOracleHeight-Unitree-G1", 64),
    ("TT-Tracking-TerrainOracleTeacher-Unitree-G1", 73),
  ],
)
def test_oracle_terrain_tracking_env_can_reset_and_step_on_cpu(
  tmp_path: Path,
  task_id: str,
  min_extra_actor_dims: int,
) -> None:
  collision_path = create_heightfield_collision_manifest(
    tmp_path / "terrain_collision.json"
  )
  bundle_dir = convert_pair(
    ConvertPairConfig(
      motion_file=create_motion_clip(tmp_path / "motion.npz"),
      terrain_file=create_ramp_obj(tmp_path / "terrain.obj"),
      terrain_collision_file=collision_path,
      output_dir=tmp_path / "converted",
      sample_name="smoke_pair",
    )
  )

  blind_env, _blind_agent_cfg = build_paired_env(
    "TT-Tracking-TerrainBlind-Unitree-G1",
    str(bundle_dir / "pair.json"),
    play=True,
    device="cpu",
    num_envs=1,
    no_terminations=True,
  )
  oracle_env, _oracle_agent_cfg = build_paired_env(
    task_id,
    str(bundle_dir / "pair.json"),
    play=True,
    device="cpu",
    num_envs=1,
    no_terminations=True,
  )
  try:
    blind_obs, _blind_extras = blind_env.reset()
    oracle_obs, _oracle_extras = oracle_env.reset()

    blind_actor_dim = blind_obs["actor"].shape[-1]
    oracle_actor_dim = oracle_obs["actor"].shape[-1]
    assert oracle_actor_dim >= blind_actor_dim + min_extra_actor_dims

    action_dim = oracle_env.unwrapped.single_action_space.shape[0]
    action = torch.zeros(
      (1, action_dim),
      dtype=torch.float32,
      device=oracle_env.device,
    )
    oracle_obs, reward, terminated, timeouts, extras = oracle_env.step(action)

    assert set(oracle_obs.keys()) == {"actor", "critic"}
    assert oracle_obs["actor"].shape[0] == 1
    assert reward.shape == (1,)
    assert terminated.shape == (1,)
    assert timeouts.shape == (1,)
    assert isinstance(extras, dict)
  finally:
    oracle_env.close()
    blind_env.close()
```

- [ ] **Step 2: Run oracle smoke test to verify current behavior**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_env_smoke.py -k oracle_terrain
```

Expected before Task 1-3 implementation: FAIL because oracle tasks are not registered. Expected after Task 1-3 implementation: PASS. If this fails after Task 1-3, inspect whether the terrain scan sensor collides with geom group configuration or observation dimensions differ from the expected ray count.

- [ ] **Step 3: If actor dimension assertion is off, lock the real dimension**

If the failure is only the extra actor dimension assertion, print dimensions with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_env_smoke.py -k oracle_terrain -vv -s
```

Then update only `min_extra_actor_dims` in `test_oracle_terrain_tracking_env_can_reset_and_step_on_cpu` to the observed lower bound:

```python
@pytest.mark.parametrize(
  ("task_id", "min_extra_actor_dims"),
  [
    ("TT-Tracking-TerrainOracleHeight-Unitree-G1", 64),
    ("TT-Tracking-TerrainOracleTeacher-Unitree-G1", 73),
  ],
)
```

Keep `TerrainOracleTeacher` at least 9 dimensions larger than `TerrainOracleHeight` because the teacher terms add three `[num_envs, 3]` tensors.

- [ ] **Step 4: Run focused env smoke tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_env_smoke.py -k "oracle_terrain or blind_terrain_tracking_env_can_reset"
```

Expected: PASS.

- [ ] **Step 5: Commit env smoke coverage**

Run:

```bash
git add tests/test_env_smoke.py
git commit -m "test: add oracle terrain tracking env smoke"
```

---

### Task 5: Document Oracle Training Commands

**Files:**
- Modify: `README.md`
- Test: `README.md` plus CLI help smoke

- [ ] **Step 1: Add oracle task usage to README**

Append this section to `README.md`:

```markdown
## Oracle Perception Tasks

This package also registers oracle perception tracking tasks for the current height-map stage:

- `TT-Tracking-TerrainOracleHeight-Unitree-G1`: blind tracking observations plus a clean `0.7 m x 0.7 m` yaw-aligned terrain height scan from `torso_link`.
- `TT-Tracking-TerrainOracleTeacher-Unitree-G1`: oracle height observations plus PHP teacher-style global tracking error terms for `torso_link`.

Both tasks reuse the blind tracking rewards, terminations, terrain pair application, and PPO defaults. They are simulation-only oracle tasks and are not deployment policies.

Train with the existing paired training entry point:

```bash
uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.train \
  --pair-manifest /abs/path/to/pair.json \
  --task TT-Tracking-TerrainOracleHeight-Unitree-G1
```

```bash
uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.train \
  --pair-manifest /abs/path/to/pair.json \
  --task TT-Tracking-TerrainOracleTeacher-Unitree-G1
```
```

- [ ] **Step 2: Run CLI help smoke**

Run:

```bash
uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.train --help
```

Expected: PASS and help output includes `--task STR`.

- [ ] **Step 3: Run focused tests after README update**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_registry.py tests/test_oracle_terrain_tracking_config.py tests/test_env_smoke.py -k "oracle or blind_terrain_tracking_env_can_reset"
```

Expected: PASS.

- [ ] **Step 4: Commit docs**

Run:

```bash
git add README.md
git commit -m "docs: document oracle terrain tracking tasks"
```

---

### Task 6: Full Verification And Training Smoke

**Files:**
- No code files should be changed in this task unless verification exposes a concrete bug.
- Test: full pytest suite and short training smoke.

- [ ] **Step 1: Run full test suite**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests
```

Expected: PASS.

- [ ] **Step 2: Prepare or select a paired manifest for training smoke**

Use an existing converted pair manifest with a collision manifest. Set:

```bash
PAIR_MANIFEST=/abs/path/to/pair.json
```

The selected `pair.json` should include `terrain_collision_file` so the default primitive box collision backend is used.

- [ ] **Step 3: Run short oracle height training smoke**

Run:

```bash
uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.train \
  --pair-manifest "${PAIR_MANIFEST}" \
  --task TT-Tracking-TerrainOracleHeight-Unitree-G1 \
  --env.scene.num-envs 1 \
  --agent.max-iterations 100 \
  --agent.logger tensorboard \
  --agent.save-interval 100 \
  --agent.run-name oracle_height_smoke
```

Expected: training starts, completes 100 iterations, and writes logs under `logs/rsl_rl/g1_oracle_terrain_tracking/`.

- [ ] **Step 4: Run short oracle teacher training smoke**

Run:

```bash
uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.train \
  --pair-manifest "${PAIR_MANIFEST}" \
  --task TT-Tracking-TerrainOracleTeacher-Unitree-G1 \
  --env.scene.num-envs 1 \
  --agent.max-iterations 100 \
  --agent.logger tensorboard \
  --agent.save-interval 100 \
  --agent.run-name oracle_teacher_smoke
```

Expected: training starts, completes 100 iterations, and writes logs under `logs/rsl_rl/g1_oracle_terrain_tracking/`.

- [ ] **Step 5: Inspect final git state**

Run:

```bash
git status --short
```

Expected: only intended logs or user-owned untracked files remain. Do not stage generated training logs.

- [ ] **Step 6: Commit any verification fixes**

If Task 6 required a code fix, run the relevant focused test again, then commit only the source/test/doc files for that fix:

```bash
git add src tests README.md
git commit -m "fix: stabilize oracle terrain tracking smoke"
```

If Task 6 required no code fix, do not create an empty commit.
