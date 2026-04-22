# TT Convert Pair Design

**Goal:** Add an upstream conversion path for `terrain_tracking/` that turns one retargeted PARC sample into one self-contained pair bundle consumable by the blind terrain tracking task.

**Status:** Approved design

## Problem

`terrain_tracking/` already knows how to consume a paired runtime bundle:

- one canonical `mjlab` motion file
- one terrain mesh
- one pair manifest that points at both files

What it does **not** do is convert upstream retargeting output into that bundle.

The current upstream artifacts already exist for validation:

- retargeted motion as `qpos`-based `.npz`
- terrain mesh as `.obj`

Those artifacts are not directly consumable by `terrain_tracking/` because:

- `terrain_tracking/` expects canonical `mjlab` tracking motion arrays
- the task runtime expects a `pair.json` manifest
- motion canonicalization must follow `mjlab`'s own body ordering and asset layout

## Scope Lock

This design is intentionally narrow.

Included:

- single-sample conversion only
- input: retargeted `qpos` `.npz` plus terrain `.obj`
- output: one self-contained pair bundle
- target runtime: `terrain_tracking/` only
- canonicalization using official `mjlab==1.3.0`

Excluded:

- holosoma WBT output
- multi-motion dataset generation
- batch scheduling or manifests of manifests
- terrain-aware observation design
- automatic geometric realignment beyond thin transform fields in `pair.json`

## Dependency Boundary

`terrain_tracking/` is a downstream `mjlab` project and must consume official `mjlab` as a package dependency.

That means:

- runtime imports come from the installed `mjlab` package in the `terrain_tracking` environment
- local `controller/mjlab` is reference-only
- we may read reference code in `controller/mjlab`, but we do not depend on it as a path dependency

This matters for motion canonicalization because `mjlab`'s tracking format stores precomputed body arrays whose indexing depends on `mjlab`'s own MuJoCo asset traversal.

## Output Contract

Each conversion produces one directory:

```text
<output_root>/<sample_name>/
  motion.npz
  terrain.obj
  pair.json
  meta.json
```

### `motion.npz`

This file is in canonical `mjlab` tracking format and contains:

- `fps`
- `joint_pos`
- `joint_vel`
- `body_pos_w`
- `body_quat_w`
- `body_lin_vel_w`
- `body_ang_vel_w`

No `joint_names` or `body_names` are required for the current `terrain_tracking/` runtime.

### `pair.json`

This file matches the existing runtime contract:

```json
{
  "motion_file": "motion.npz",
  "terrain_file": "terrain.obj",
  "terrain_translation": [0.0, 0.0, 0.0],
  "terrain_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
  "terrain_scale": [1.0, 1.0, 1.0]
}
```

For phase 1, the default transform is identity unless a caller explicitly overrides it.

### `meta.json`

This file is for traceability only and is not consumed by runtime.

It records:

- source retargeted `.npz`
- source terrain `.obj`
- sample name
- input fps
- output fps
- frame count
- converter version

## Input Contract

The upstream retargeted `.npz` is expected to contain at least:

- `qpos`
- `fps`

Phase 1 ignores optional fields like `human_joints` and `cost` for runtime output, though they may be copied into metadata if useful.

The expected `qpos` layout is:

- columns `0:3` root translation
- columns `3:7` root quaternion
- columns `7:` actuated joint positions

## Data Flow

Conversion proceeds in four layers.

### 1. Retarget Input Parsing

Read the retargeted `.npz` and normalize it into an intermediate in-memory clip:

- root position
- root orientation
- actuated joint positions
- input fps
- frame count

This layer does not touch terrain and does not know about output packaging.

### 2. Canonical `mjlab` Motion Emission

Replay the normalized clip through official `mjlab` G1 assets.

For each frame:

- write root state
- write joint state
- advance the sim / scene once
- read back canonical `joint_pos`
- read back canonical `joint_vel`
- read back canonical body poses and velocities

This is the critical step. It ensures the stored body arrays are generated using `mjlab`'s own asset ordering rather than a foreign simulator's ordering.

### 3. Pair Bundle Packaging

Create the output directory and write:

- `motion.npz`
- copied `terrain.obj`
- `pair.json`
- `meta.json`

The bundle is self-contained and does not depend on `/tmp/parc_process_workspace/...` remaining alive.

### 4. Runtime Verification

The produced `pair.json` should work directly with:

- `terrain_tracking.tasks.blind_terrain_tracking.scripts.play`
- `terrain_tracking.tasks.blind_terrain_tracking.scripts.train`

No additional conversion step should be needed.

## Code Layout

Core logic lives in a reusable package module:

```text
src/terrain_tracking/data_conversion/
  __init__.py
  retarget_input.py
  mjlab_motion.py
  pair_bundle.py
  convert_pair.py
```

Thin CLI entrypoint:

```text
src/terrain_tracking/tasks/blind_terrain_tracking/scripts/convert.py
```

Thin shell wrapper:

```text
scripts/convert_pair.sh
```

This keeps task-facing CLI placement consistent with existing `train.py`, `play.py`, and `evaluate.py`, while keeping the conversion logic testable and reusable.

## Why Not Reuse Holosoma's Output Schema

Holosoma's `convert_data_format_mj.py` is a useful reference for:

- parsing `qpos`
- time resampling
- replaying motions through MuJoCo to derive body trajectories

But its final output schema is for holosoma WBT, not `terrain_tracking/`.

Differences:

- holosoma expects `joint_names` and `body_names`
- holosoma stores root DOFs inside `joint_pos` and `joint_vel`
- `terrain_tracking/` expects canonical `mjlab` tracking arrays and a separate `pair.json`

So the correct approach is:

- borrow the conversion pattern
- target `mjlab`/`terrain_tracking` output
- keep the resulting script in `terrain_tracking/`

## Validation Samples

Phase 1 validation uses these existing artifacts:

- `/tmp/parc_process_workspace/retargeted/platform_001_original.npz`
- `/tmp/parc_process_workspace/workspace/platform_001/multi_boxes.obj`
- `/tmp/parc_process_workspace/retargeted/mid_blocks_004_dm_original.npz`
- `/tmp/parc_process_workspace/workspace/mid_blocks_004_dm/multi_boxes.obj`

These are enough to validate:

- conversion correctness
- pair bundle writing
- runtime loading
- short training initialization

## Verification Plan

Automated verification:

- parser unit tests
- bundle writer unit tests
- conversion integration test on a tiny synthetic clip
- runtime contract test using `PairManifest.load()`

Manual verification:

- run convert on `platform_001`
- run `play` on generated `pair.json`
- run convert on `mid_blocks_004_dm`
- run a short training smoke test on one generated pair

## Success Criteria

Phase 1 is complete when:

- one CLI command converts an upstream retargeted sample into a self-contained pair bundle
- the bundle loads directly in `terrain_tracking`
- `play` succeeds on at least one real PARC-derived pair
- a short training smoke run initializes and collects rollouts without format errors
