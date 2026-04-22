# TT Blind Terrain Tracking Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `terrain_tracking/` as a downstream `mjlab==1.3.0` extension package that supports single paired motion blind terrain tracking for Unitree G1 via `TT-Tracking-TerrainBlind-Unitree-G1`.

**Architecture:** Reuse `mjlab`'s native single-motion tracking stack for PPO, motion command, reward, and termination logic. Add only three project-specific layers: pair-manifest parsing, paired terrain mesh injection into the MuJoCo scene, and a terrain-blind G1 task configuration that tightens RSI/randomization so the motion-terrain pairing is not destroyed at reset.

**Tech Stack:** Python, `mjlab[cu128]==1.3.0`, `trimesh`, `uv`, `pytest`, MuJoCo via `mjlab`.

---

## Scope Lock

- Project root: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking`
- Package name: `terrain_tracking`
- Task family directory: `src/terrain_tracking/tasks/blind_terrain_tracking/`
- Task IDs:
  - `TT-Tracking-TerrainBlind-Unitree-G1`
  - `TT-Tracking-TerrainBlind-Unitree-G1-No-State-Estimation`
- `mjlab` dependency strategy:
  - depend on `mjlab[cu128]==1.3.0`
  - do not depend on `controller/mjlab` as a path dependency
  - do not modify `mjlab` source as part of Phase 1
- Phase 1 scope:
  - single paired motion only
  - blind terrain tracking only
  - PPO only
  - terrain treated as continuous ground
  - root `scripts/*.sh` are convenience wrappers only and carry no task semantics

## Pair Manifest Contract

Phase 1 runtime input is one pair manifest:

```json
{
  "motion_file": "motion.npz",
  "terrain_file": "terrain.obj",
  "terrain_translation": [0.0, 0.0, 0.0],
  "terrain_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
  "terrain_scale": [1.0, 1.0, 1.0]
}
```

Assumptions:

- `motion.npz` is already in canonical `mjlab` tracking format.
- motion and terrain are already aligned by the upstream conversion pipeline.
- training-side transform fields are only thin overrides, not a full alignment system.

## Final Repository Layout

```text
terrain_tracking/
  pyproject.toml
  README.md
  docs/
    plans/
      2026-04-21-tt-blind-terrain-tracking-phase1.md
  scripts/
    train.sh
    play.sh
  src/terrain_tracking/
    __init__.py
    _mjlab_tasks.py
    runtime/
      pair_manifest.py
      apply_pair.py
    scene/
      paired_mesh_spec.py
    tasks/
      blind_terrain_tracking/
        __init__.py
        config/
          g1/
            __init__.py
            env_cfgs.py
            rl_cfg.py
        scripts/
          train.py
          play.py
          evaluate.py
  tests/
    test_pair_manifest.py
    test_paired_mesh_spec.py
    test_env_smoke.py
    test_registry.py
```

## Reference Reuse from `mjlab`

Phase 1 should directly reuse these upstream components:

- base tracking env cfg:
  `/home/humanoid/Projects/Junsong_WU/learning/locomotion/controller/mjlab/src/mjlab/tasks/tracking/tracking_env_cfg.py`
- G1 flat tracking cfg:
  `/home/humanoid/Projects/Junsong_WU/learning/locomotion/controller/mjlab/src/mjlab/tasks/tracking/config/g1/env_cfgs.py`
- motion command:
  `/home/humanoid/Projects/Junsong_WU/learning/locomotion/controller/mjlab/src/mjlab/tasks/tracking/mdp/commands.py`
- tracking runner:
  `/home/humanoid/Projects/Junsong_WU/learning/locomotion/controller/mjlab/src/mjlab/tasks/tracking/config/g1/__init__.py`
- scene callback hook:
  `/home/humanoid/Projects/Junsong_WU/learning/locomotion/controller/mjlab/src/mjlab/scene/scene.py`

## Task 1: Scaffold the Downstream Package

**Files:**
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/pyproject.toml`
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/README.md`
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/__init__.py`
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/_mjlab_tasks.py`
- Test: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/tests/test_registry.py`

**Step 1: Write the failing registry test**

Write `tests/test_registry.py` to assert:

- the package exposes a `mjlab.tasks` entry point
- importing the package registers `TT-Tracking-TerrainBlind-Unitree-G1`

**Step 2: Run test to verify it fails**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
uv run pytest -q tests/test_registry.py
```

Expected: FAIL because the package metadata and task registration do not exist yet.

**Step 3: Write minimal implementation**

Implement:

- `pyproject.toml` with:
  - project name `terrain-tracking`
  - dependencies `mjlab[cu128]==1.3.0` and `trimesh>=4.8`
  - entry point:
    `terrain_tracking = "terrain_tracking._mjlab_tasks"`
- `__init__.py`
- `_mjlab_tasks.py` stub that imports the task config module

**Step 4: Run test to verify it passes**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
uv run pytest -q tests/test_registry.py
```

Expected: PASS

## Task 2: Implement Pair Manifest Parsing

**Files:**
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/runtime/pair_manifest.py`
- Test: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/tests/test_pair_manifest.py`

**Step 1: Write the failing manifest tests**

Cover:

- valid manifest loads
- relative `motion_file` and `terrain_file` resolve relative to manifest location
- missing required keys fail with clear error
- optional transform fields default correctly

**Step 2: Run test to verify it fails**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
uv run pytest -q tests/test_pair_manifest.py
```

Expected: FAIL because the runtime parser does not exist.

**Step 3: Write minimal implementation**

Implement:

- a frozen dataclass `PairManifest`
- JSON loading from file
- path normalization to absolute paths
- defaults for translation, quaternion, and scale
- validation that the manifest points to existing files

**Step 4: Run test to verify it passes**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
uv run pytest -q tests/test_pair_manifest.py
```

Expected: PASS

## Task 3: Implement Paired Terrain Mesh Injection

**Files:**
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/scene/paired_mesh_spec.py`
- Test: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/tests/test_paired_mesh_spec.py`

**Step 1: Write the failing scene mesh tests**

Cover:

- `.obj` can be loaded through `trimesh`
- mesh vertices and faces are converted to MuJoCo `uservert/userface` input
- one mesh copy is created per environment origin
- translation/rotation/scale are applied before duplication

**Step 2: Run test to verify it fails**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
uv run pytest -q tests/test_paired_mesh_spec.py
```

Expected: FAIL because terrain mesh injection utilities do not exist.

**Step 3: Write minimal implementation**

Implement utilities that:

- load the terrain with `trimesh`
- coerce scenes into one mesh if needed
- apply manifest transform
- duplicate the mesh at each env origin
- return a `spec_fn` callback that adds mesh geoms to the scene spec

**Step 4: Run test to verify it passes**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
uv run pytest -q tests/test_paired_mesh_spec.py
```

Expected: PASS

## Task 4: Bind a Pair Manifest into an Env Config

**Files:**
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/runtime/apply_pair.py`
- Modify: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/scene/paired_mesh_spec.py`
- Test: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/tests/test_env_smoke.py`

**Step 1: Write the failing env smoke test**

Build a tiny synthetic pair:

- a minimal valid `motion.npz`
- a tiny continuous-ground `.obj`
- a manifest pointing to both

Then assert:

- env config accepts the pair
- scene can compile
- env can reset and step once

**Step 2: Run test to verify it fails**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
uv run pytest -q tests/test_env_smoke.py
```

Expected: FAIL because env config cannot yet accept a pair manifest.

**Step 3: Write minimal implementation**

Implement `apply_pair.py` to:

- parse a manifest path
- set `motion_cmd.motion_file`
- attach the scene `spec_fn`
- leave task semantics in Python, not in shell wrappers

**Step 4: Run test to verify it passes**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
uv run pytest -q tests/test_env_smoke.py
```

Expected: PASS for compile/reset/step.

## Task 5: Build the G1 Blind Terrain Tracking Task Config

**Files:**
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/tasks/blind_terrain_tracking/__init__.py`
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/tasks/blind_terrain_tracking/config/g1/__init__.py`
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/tasks/blind_terrain_tracking/config/g1/env_cfgs.py`
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/tasks/blind_terrain_tracking/config/g1/rl_cfg.py`
- Modify: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/_mjlab_tasks.py`
- Test: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/tests/test_registry.py`

**Step 1: Extend the failing registry test**

Add checks for:

- `TT-Tracking-TerrainBlind-Unitree-G1`
- `TT-Tracking-TerrainBlind-Unitree-G1-No-State-Estimation`

**Step 2: Run test to verify it fails**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
uv run pytest -q tests/test_registry.py
```

Expected: FAIL because the task configs are not yet implemented.

**Step 3: Write minimal implementation**

Build the env cfg by reusing `mjlab` tracking pieces and overriding:

- task id names
- G1 robot config
- blind terrain tracking defaults
- RSI tightening:
  - `x/y/yaw` pose randomization set to zero
  - `z/roll/pitch` set to zero or near zero
  - `velocity_range` heavily reduced
  - `push_robot` removed
- `No-State-Estimation` variant mirrors upstream style

Keep observations terrain-blind:

- no terrain sensor
- no terrain scan observation
- no terrain-aware reward or termination

**Step 4: Run test to verify it passes**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
uv run pytest -q tests/test_registry.py
```

Expected: PASS

## Task 6: Add Task-Local Python Entry Points

**Files:**
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/tasks/blind_terrain_tracking/scripts/train.py`
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/tasks/blind_terrain_tracking/scripts/play.py`
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/tasks/blind_terrain_tracking/scripts/evaluate.py`
- Modify: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/tests/test_env_smoke.py`

**Step 1: Write a failing CLI smoke test or module import test**

Cover:

- train script accepts `--pair-manifest`
- play script accepts `--pair-manifest`
- scripts can build an env cfg with the registered task id

**Step 2: Run test to verify it fails**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
uv run pytest -q tests/test_env_smoke.py
```

Expected: FAIL because task-local entrypoints do not exist.

**Step 3: Write minimal implementation**

Implement:

- `train.py` to parse task id and pair manifest, then launch `mjlab` training
- `play.py` to load pair manifest and checkpoint for playback
- `evaluate.py` as a thin reserved entrypoint following the same pattern

All runtime semantics stay here, not in root shell scripts.

**Step 4: Run test to verify it passes**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
uv run pytest -q tests/test_env_smoke.py
```

Expected: PASS

## Task 7: Add Root Convenience Wrappers

**Files:**
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/scripts/train.sh`
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/scripts/play.sh`
- Modify: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/README.md`

**Step 1: Write a simple usage check**

Document and verify that:

- `scripts/train.sh` calls the task-local Python train script
- `scripts/play.sh` calls the task-local Python play script
- wrappers do not contain tracking semantics beyond argument forwarding and common defaults

**Step 2: Manually verify wrapper behavior**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
bash scripts/train.sh --help
bash scripts/play.sh --help
```

Expected: each wrapper reaches the corresponding Python entrypoint help text.

**Step 3: Write minimal implementation**

Implement wrappers that:

- use `uv run python -m ...`
- forward arguments verbatim
- optionally set a default task id

Do not encode pair semantics, terrain semantics, or hardcoded experiment logic in shell.

## Task 8: Full Verification

**Files:**
- Modify as needed from previous tasks only
- Test: all tests above

**Step 1: Run unit and smoke tests**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
uv run pytest -q tests
```

Expected: PASS

**Step 2: Run a short smoke training**

Run something equivalent to:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.train \
  --task TT-Tracking-TerrainBlind-Unitree-G1 \
  --pair-manifest /abs/path/to/pair.json
```

Expected:

- env builds
- motion file loads
- terrain mesh is injected
- PPO starts stepping without immediate crash

**Step 3: Run play smoke test**

Run something equivalent to:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.play \
  --task TT-Tracking-TerrainBlind-Unitree-G1 \
  --pair-manifest /abs/path/to/pair.json
```

Expected:

- robot and terrain both appear
- tracking runs in the paired terrain scene

## Phase 1 Completion Criteria

Phase 1 is complete when all of the following are true:

- `terrain_tracking/` installs as a downstream `mjlab` extension package
- `TT-Tracking-TerrainBlind-Unitree-G1` and `TT-Tracking-TerrainBlind-Unitree-G1-No-State-Estimation` register successfully
- one pair manifest can drive both motion loading and terrain injection
- the task remains blind to terrain in actor and critic observations
- the environment can compile, reset, and step
- a short PPO smoke training can run on one real paired motion without immediate instability
- `scripts/train.sh` and `scripts/play.sh` only wrap the Python entrypoints and carry no extra semantics

## Explicitly Deferred to Later Phases

- multi-motion datasets
- general tracking datasets
- custom multi-motion sampling
- terrain-aware observations
- terrain-aware reward shaping
- terrain-aware terminations
- object or scene support beyond one paired terrain mesh
- replacing the temporary continuous-ground assumption with strict no-plane geometry
