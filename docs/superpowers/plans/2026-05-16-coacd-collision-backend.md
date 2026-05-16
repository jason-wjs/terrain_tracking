# CoACD Collision Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `collision_backend="coacd"` for single-pair terrain tracking so CoACD collision can be compared with the existing `primitive_boxes` backend.

**Architecture:** Add a focused `coacd_mesh_spec.py` scene module that loads the pair visual mesh, applies the existing pair transform semantics, decomposes the transformed mesh into convex hulls, caches hull arrays, and injects visual plus collision geoms into `mujoco.MjSpec`. Wire the new backend through `apply_pair_manifest_to_env_cfg()` without changing the default backend.

**Tech Stack:** Python 3.10+, MuJoCo `MjSpec`, `trimesh`, `numpy`, optional `coacd`, `pytest`, `uv`.

---

## Files

- Create: `src/terrain_tracking/scene/coacd_mesh_spec.py`
  - Owns CoACD options, cache key/path, disk cache IO, hull sanitization, optional `coacd` import, and `make_coacd_mesh_spec_fn()`.
- Modify: `src/terrain_tracking/runtime/apply_pair.py`
  - Accepts `collision_backend="coacd"` and routes to `make_coacd_mesh_spec_fn()`.
- Modify: `src/terrain_tracking/scene/__init__.py`
  - Exports CoACD scene helpers.
- Modify: `pyproject.toml`
  - Adds `coacd>=1.0.7`.
- Modify: `uv.lock`
  - Lockfile update after dependency change.
- Create: `tests/test_coacd_mesh_spec.py`
  - Focused unit tests for visual/collision geom separation, caching, missing dependency errors, and degenerate hull handling.
- Modify: `tests/test_env_smoke.py`
  - Adds a small runtime wiring test that `collision_backend="coacd"` is accepted and calls the CoACD spec factory.

## Baseline

Before implementation, the isolated worktree full test command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests
```

returns `48 passed, 1 failed`. The failing test is unrelated:

```text
tests/test_cli_help.py::test_train_shell_uses_mjlab_env_spacing_flag
```

Focused CoACD tests must pass. The full suite may still report that baseline failure.

### Task 1: Add CoACD Tests

**Files:**
- Create: `tests/test_coacd_mesh_spec.py`
- Modify: `tests/test_env_smoke.py`

- [ ] **Step 1: Write failing scene tests**

Create `tests/test_coacd_mesh_spec.py` with tests that expect:

```python
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from terrain_tracking.runtime.pair_manifest import PairManifest
from terrain_tracking.scene import coacd_mesh_spec
from terrain_tracking.scene.coacd_mesh_spec import (
  CoacdCollisionOptions,
  make_coacd_mesh_spec_fn,
)
from tests.helpers import create_motion_clip, create_quad_obj


def _manifest(tmp_path: Path) -> PairManifest:
  motion_path = create_motion_clip(tmp_path / "motion.npz")
  terrain_path = create_quad_obj(tmp_path / "terrain.obj")
  return PairManifest(motion_file=motion_path.resolve(), terrain_file=terrain_path.resolve())


def _tetra_parts() -> list[tuple[np.ndarray, np.ndarray]]:
  return [
    (
      np.array(
        [
          [0.0, 0.0, 0.0],
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 0.4],
        ],
        dtype=np.float32,
      ),
      np.array(
        [
          [0, 1, 2],
          [0, 1, 3],
          [1, 2, 3],
          [2, 0, 3],
        ],
        dtype=np.int32,
      ),
    )
  ]


def test_coacd_spec_adds_visual_mesh_and_collision_hulls(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  manifest = _manifest(tmp_path)
  monkeypatch.setattr(coacd_mesh_spec, "_load_or_compute_coacd_parts", lambda pair, options: _tetra_parts())
  spec_fn = make_coacd_mesh_spec_fn(
    manifest,
    num_envs=2,
    env_spacing=2.0,
    options=CoacdCollisionOptions(cache_dir=tmp_path / "cache"),
  )

  spec = mujoco.MjSpec()
  spec_fn(spec)

  assert len(spec.meshes) == 2
  paired_bodies = [body for body in spec.worldbody.bodies if body.name.startswith("paired_terrain_")]
  assert len(paired_bodies) == 2
  for body in paired_bodies:
    assert len(body.geoms) == 2
    visual, hull = body.geoms
    assert visual.type == mujoco.mjtGeom.mjGEOM_MESH
    assert visual.contype == 0
    assert visual.conaffinity == 0
    assert hull.type == mujoco.mjtGeom.mjGEOM_MESH
    assert hull.contype == 1
    assert hull.conaffinity == 1
    assert hull.margin == 0.0
    assert hull.gap == 0.0


def test_coacd_disk_cache_reuses_saved_parts(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  manifest = _manifest(tmp_path)
  options = CoacdCollisionOptions(cache_dir=tmp_path / "cache")
  calls = 0

  def fake_run(mesh, run_options):
    nonlocal calls
    calls += 1
    return _tetra_parts()

  monkeypatch.setattr(coacd_mesh_spec, "_run_coacd_decomposition", fake_run)
  first = coacd_mesh_spec._load_or_compute_coacd_parts(manifest, options)
  coacd_mesh_spec._COACD_PARTS_CACHE.clear()
  monkeypatch.setattr(
    coacd_mesh_spec,
    "_run_coacd_decomposition",
    lambda mesh, run_options: pytest.fail("cache miss"),
  )
  second = coacd_mesh_spec._load_or_compute_coacd_parts(manifest, options)

  assert calls == 1
  np.testing.assert_allclose(first[0][0], second[0][0])
  np.testing.assert_array_equal(first[0][1], second[0][1])


def test_missing_coacd_package_reports_backend_name(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  manifest = _manifest(tmp_path)
  mesh = coacd_mesh_spec._load_transformed_trimesh(manifest)
  monkeypatch.setattr(coacd_mesh_spec, "_import_coacd", lambda: (_ for _ in ()).throw(ImportError("missing")))

  with pytest.raises(RuntimeError, match="collision_backend='coacd'.*coacd"):
    coacd_mesh_spec._run_coacd_decomposition(mesh, CoacdCollisionOptions(cache_dir=tmp_path / "cache"))


def test_sanitize_coacd_parts_rejects_all_degenerate_parts() -> None:
  with pytest.raises(ValueError, match="no valid 3D hulls"):
    coacd_mesh_spec._sanitize_coacd_parts(
      [(np.zeros((3, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.int32))],
      terrain_tag="degenerate",
    )
```

- [ ] **Step 2: Write failing runtime wiring test**

Append a test to `tests/test_env_smoke.py`:

```python
def test_apply_pair_manifest_accepts_coacd_backend(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  motion_path = create_motion_clip(tmp_path / "motion.npz")
  terrain_path = create_ramp_obj(tmp_path / "terrain.obj")
  manifest_path = create_pair_manifest(
    tmp_path / "pair.json",
    motion_file=motion_path.name,
    terrain_file=terrain_path.name,
  )
  calls = []

  def fake_make_coacd_mesh_spec_fn(pair, *, num_envs, env_spacing):
    calls.append((pair, num_envs, env_spacing))
    return lambda spec: None

  monkeypatch.setattr(
    "terrain_tracking.runtime.apply_pair.make_coacd_mesh_spec_fn",
    fake_make_coacd_mesh_spec_fn,
  )

  cfg = unitree_g1_blind_terrain_tracking_env_cfg()
  apply_pair_manifest_to_env_cfg(cfg, manifest_path, collision_backend="coacd")
  cfg.scene.num_envs = 3
  cfg.scene.env_spacing = 4.0

  assert cfg.scene.spec_fn is not None
  cfg.scene.spec_fn(mujoco.MjSpec())

  assert len(calls) == 1
  pair, num_envs, env_spacing = calls[0]
  assert pair.terrain_file == terrain_path.resolve()
  assert num_envs == 3
  assert env_spacing == 4.0
  assert cfg.sim.nconmax == 256
  assert cfg.sim.njmax == 512
```

- [ ] **Step 3: Verify tests fail for missing implementation**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/test_coacd_mesh_spec.py tests/test_env_smoke.py::test_apply_pair_manifest_accepts_coacd_backend
```

Expected: import or backend errors because `coacd_mesh_spec.py` and the `coacd` branch do not exist yet.

### Task 2: Implement CoACD Scene Module

**Files:**
- Create: `src/terrain_tracking/scene/coacd_mesh_spec.py`
- Modify: `src/terrain_tracking/scene/__init__.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Add dependency**

Add this dependency to `pyproject.toml`:

```toml
  "coacd>=1.0.7",
```

Run:

```bash
uv lock
```

Expected: `uv.lock` includes the `coacd` package.

- [ ] **Step 2: Implement `coacd_mesh_spec.py`**

Create `src/terrain_tracking/scene/coacd_mesh_spec.py` with:

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import hashlib
import importlib
import os
import uuid

import mujoco
import numpy as np
import trimesh

from terrain_tracking.runtime.pair_manifest import PairManifest
from terrain_tracking.scene.paired_mesh_spec import (
  _load_mesh_arrays,
  compute_env_origins_grid,
)
```

Implement `CoacdCollisionOptions`, cache IO, `_sanitize_coacd_parts()`,
`_run_coacd_decomposition()`, `_load_or_compute_coacd_parts()`, and
`make_coacd_mesh_spec_fn()` according to the design spec.

- [ ] **Step 3: Export helpers**

Update `src/terrain_tracking/scene/__init__.py` to import and include:

```python
from .coacd_mesh_spec import CoacdCollisionOptions, make_coacd_mesh_spec_fn
```

and add both names to `__all__`.

- [ ] **Step 4: Verify scene tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/test_coacd_mesh_spec.py
```

Expected: all tests in `tests/test_coacd_mesh_spec.py` pass.

### Task 3: Wire Runtime Backend

**Files:**
- Modify: `src/terrain_tracking/runtime/apply_pair.py`

- [ ] **Step 1: Add import and backend validation**

Import:

```python
from terrain_tracking.scene.coacd_mesh_spec import make_coacd_mesh_spec_fn
```

Change valid backend set to:

```python
{"primitive_boxes", "hfield", "mesh", "coacd"}
```

Update the error text to mention `coacd`.

- [ ] **Step 2: Route `coacd` in `pair_spec_fn`**

In `pair_spec_fn`, handle:

```python
if collision_backend == "coacd":
  make_coacd_mesh_spec_fn(
    pair,
    num_envs=cfg.scene.num_envs,
    env_spacing=cfg.scene.env_spacing,
  )(spec)
elif pair.terrain_collision_file is not None and collision_backend == "hfield":
  ...
else:
  ...
```

Set minimum contact buffers for `coacd`:

```python
if pair.terrain_collision_file is not None or collision_backend == "coacd":
  _set_min_sim_contact_buffers(cfg, nconmax=256, njmax=512)
```

- [ ] **Step 3: Verify runtime wiring test**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/test_env_smoke.py::test_apply_pair_manifest_accepts_coacd_backend
```

Expected: the test passes.

### Task 4: Focused Verification and Commit

**Files:**
- All changed files.

- [ ] **Step 1: Run focused tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/test_coacd_mesh_spec.py tests/test_env_smoke.py::test_apply_pair_manifest_accepts_coacd_backend
```

Expected: all focused tests pass.

- [ ] **Step 2: Run related scene/runtime tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/test_paired_mesh_spec.py tests/test_env_smoke.py
```

Expected: related tests pass unless the pre-existing CLI shell failure appears only in the full suite.

- [ ] **Step 3: Run full suite**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests
```

Expected: CoACD tests pass. If `tests/test_cli_help.py::test_train_shell_uses_mjlab_env_spacing_flag` still fails, record it as the pre-existing baseline failure.

- [ ] **Step 4: Commit implementation**

Run:

```bash
git add pyproject.toml uv.lock src/terrain_tracking/runtime/apply_pair.py src/terrain_tracking/scene/__init__.py src/terrain_tracking/scene/coacd_mesh_spec.py tests/test_coacd_mesh_spec.py tests/test_env_smoke.py
git commit -m "feat: add coacd collision backend"
```

Expected: implementation commit is created on `feature/coacd-collision-backend`.

### Task 5: Push Branch

**Files:**
- Git metadata only.

- [ ] **Step 1: Push feature branch**

Run:

```bash
git push
```

Expected: `origin/feature/coacd-collision-backend` contains the spec, plan, and implementation commits.
