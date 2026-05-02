# Terrain Collision And Scaling Progress Summary

Date: 2026-04-28

This note summarizes the terrain-related debugging done so far and records the current design direction for the next terrain_tracking iteration.

## Problem Statement

terrain_tracking currently has two separate terrain problems:

1. Collision backend correctness.
   - Direct OBJ mesh collision is not valid for PARC multi-box terrain because MuJoCo mesh collision uses a convex hull style collision model for mesh geoms. The visual OBJ can look correct while collision fills or bridges concavities.
   - The later hfield backend avoids mesh convex-hull filling, but diagnostics show MuJoCo-Warp hfield contact changes early foot support dynamics enough to cause pathological tracking failure.

2. Geometry scaling and env reuse.
   - Current pair terrain injection duplicates terrain geoms once per env.
   - With many envs, this scales as `num_envs * terrain_geoms_per_pair`.
   - For primitive box decomposition, this becomes unacceptable without a tile/origin sharing design.

These two issues are independent. Fixing one does not automatically fix the other.

## Current terrain_tracking Path

The current downstream pair path is:

- `pair.json` points to:
  - `motion_file`
  - legacy `terrain_file`
  - optional `terrain_collision_file`
  - optional `terrain_visual_file`
- `apply_pair_manifest_to_env_cfg()` installs a `scene.spec_fn`.
- If `terrain_collision_file` is present, `make_heightfield_spec_fn()` injects MuJoCo hfield collision.
- Otherwise `make_paired_mesh_spec_fn()` injects OBJ mesh collision.
- Both mesh and hfield implementations currently duplicate terrain geoms per env using a grid from `compute_env_origins_grid()`.

Relevant files:

- `src/terrain_tracking/runtime/apply_pair.py`
- `src/terrain_tracking/runtime/terrain_collision.py`
- `src/terrain_tracking/scene/heightfield_spec.py`
- `src/terrain_tracking/scene/paired_mesh_spec.py`
- `src/terrain_tracking/tasks/blind_terrain_tracking/config/g1/env_cfgs.py`

## What We Verified

### OBJ Collision Is Not A Valid Backend

PARC multi-box terrain exported as one non-convex OBJ is valid as visual/debug geometry but unsafe as collision geometry.

Reason:

- MuJoCo mesh geom collision does not preserve arbitrary non-convex multi-box surfaces as authored.
- Concavities, gaps, low platforms, and side details can be filled or bridged by the collision hull.
- This can make reset and early motion collide with invisible collision surfaces even if visualization looks correct.

Decision:

- Keep `multi_boxes.obj` for upstream retargeting and downstream visualization/debug.
- Do not use it as terrain_tracking collision for training.

### HField Solves Convex-Hull Filling But Introduces A Different Training Failure

The upstream exporter now emits:

- `terrain_hf.npy`
- `terrain_collision.json`

The downstream hfield backend can load this data and display/compile a MuJoCo hfield. It also fixed the earlier high-platform placement issue after correcting axis mapping and cell-centered conversion.

However, latest training still collapsed:

- Run: `logs/rsl_rl/tt_single_pair_platform_001/2026-04-27_19-48-10_platform_001_g1_blind_NoPlaneCollision_start_n8192_it10000`
- `Train/mean_episode_length` converged to about `7.0`
- `Policy/mean_std` collapsed to about `0.079`
- Main termination was `Episode_Termination/ee_body_pos`

The failure reproduces without PPO:

- Same motion.
- `sampling_mode=start`.
- No reset randomization.
- No events.
- No terminations.
- Zero action rollout only.

Observed zero-action behavior:

- Plane backend exceeds the `0.25m` z-error threshold around state 10.
- Hfield backend exceeds or approaches the threshold around state 6-7.

Critical control experiment:

- Replacing the real `terrain_hf.npy` with an all-zero hfield produced the same early hfield behavior as the real platform hfield.
- A same-footprint primitive `box` flat platform matched the plane behavior instead of the hfield behavior.

Conclusion:

- The early step-7 pathology is not caused by PARC height values, hf transposition, scale, env spacing, or the high platform itself.
- The likely culprit is the MuJoCo-Warp hfield collision/contact path under the current G1 foot collision and tracking setup.

Decision:

- Keep hfield support as a debug/compatibility backend.
- Do not use hfield as the default training collision backend for PARC platform/block data.

### Env Spacing Overlap Was Real But Not The Final Root Cause

Earlier `env_spacing=2.0` caused terrain overlap when each env received a full hfield terrain copy.

This was fixed by:

- Adding an env spacing guard for hfield terrain footprint.
- Adding play/train support for larger spacing such as `12.0`.

After this fix, multi-env visual overlap was resolved, but the step-7 training pathology remained.

Conclusion:

- Env overlap was a real bug.
- It is not sufficient to make hfield training work.

## InstinctMJ Comparison

InstinctMJ's terrain path provides a better scaling model.

It uses a terrain generator/tile model:

- Generate a finite terrain tile grid, for example `7 x 7 = 49` tiles.
- Store `terrain_origins[num_rows, num_cols, 3]`.
- Assign each env one origin from that table.
- Many envs can share the same tile origin.
- Because mjlab/MuJoCo-Warp uses `nworld = num_envs`, envs in different worlds do not physically interact even when their assigned origin is identical.

Important distinction:

- InstinctMJ's model solves terrain geometry scaling and env reuse.
- It does not by itself solve collision backend correctness.

Relevant InstinctMJ files:

- `/home/humanoid/Projects/Junsong_WU/learning/locomotion/instinct/InstinctMJ/src/instinct_mj/terrains/terrain_importer.py`
- `/home/humanoid/Projects/Junsong_WU/learning/locomotion/instinct/InstinctMJ/src/instinct_mj/terrains/terrain_generator.py`
- `/home/humanoid/Projects/Junsong_WU/learning/locomotion/instinct/InstinctMJ/src/instinct_mj/motion_reference/motion_files/terrain_motion.py`

Key idea to port:

```text
terrain geoms scale with number of unique terrain tiles,
not with number of envs.
```

For current single-pair training:

```text
unique terrain tiles = 1
num_envs = 8192
all env origins can point to the same tile origin
```

This is valid because each env is a separate MuJoCo-Warp world.

## Recommended Next Architecture

The next terrain_tracking terrain path should combine:

1. Primitive box collision backend.
2. mjlab generator/tile style terrain reuse.

Target design:

```text
terrain_collision.json
  -> load PARC hf and metadata
  -> decompose piecewise-constant heights into primitive boxes
  -> generate one terrain tile, not one terrain copy per env
  -> use mjlab TerrainGenerator/TerrainEntity origin assignment
  -> assign all single-pair envs to the same terrain origin
```

For future multi-motion/multi-terrain metadata:

```text
unique terrain manifests -> unique terrain tiles
motion/pair metadata -> matching tile origin candidates
env reset -> choose an origin compatible with sampled motion
```

## Primitive Box Backend Scope

Primitive boxes are appropriate for PARC terrains that are piecewise-constant, box-like, platform-like, stair-like, or block-like.

They are not a universal heightfield replacement for continuous noisy terrain.

Conversion should include complexity checks:

- hf shape
- unique height count after quantization
- nonzero cell count
- rectangle-merged box count
- box count per tile
- unsupported complexity status when thresholds are exceeded

For `platform_001`, current data is simple:

- `terrain_hf.npy` shape: `(20, 32)`
- heights: `0.0` and `0.6`
- nonzero cells: `15`

This is a strong candidate for primitive box collision.

## Open Risks

- Cell-center versus cell-corner convention must remain exact. This is the most likely source of terrain/motion misalignment.
- Rectangle merge must not create gaps or shrink support surfaces.
- Box count must be bounded before enabling large `num_envs`.
- Current mjlab Scene uses `TerrainEntity` for env origin assignment. terrain_tracking should prefer native `TerrainGeneratorCfg`/`SubTerrainCfg` rather than custom per-env `spec_fn` terrain duplication.
- Hfield backend should remain available for diagnostics but should not be the default for this PARC platform training path.

## Current Decision

Do not continue trying to make the current per-env hfield backend train.

Next implementation should:

1. Add a primitive box terrain backend from `terrain_collision.json`.
2. Route pair terrain through a tile/generator origin model, not per-env geom duplication.
3. Use single-tile shared-origin training for current single-pair experiments.
4. Keep OBJ visual and hfield debug support as non-default compatibility paths.

Implementation status:

- `primitive_boxes` is the intended default runtime backend for pair manifests
  with `terrain_collision_file`.
- `hfield` remains an explicit debug backend.
- `mesh` remains a legacy fallback for pair manifests without collision
  manifests or for explicit compatibility checks.
- The scaling target is still that terrain geoms scale with unique terrain tiles,
  not with `num_envs`.
