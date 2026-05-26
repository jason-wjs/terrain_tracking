# General Terrain-Aware Teacher Design

This design adds a general teacher task for training one terrain-aware
controller across many motion-terrain pairs. The existing single-pair
`Pair manifest` train/play path remains intact.

## Goals

- Train a teacher controller over a large `Pair dataset` of PARC motion-terrain
  pairs, with later support for OmniRetarget pairs.
- Preserve the invariant that each reset samples one `Pair ID`, and that ID
  selects both the motion clip and terrain tile.
- Compile a static full `Terrain tile bank` once at environment startup.
- Reuse the current oracle teacher observations and rewards where possible.
- Keep student distillation out of scope for this phase.

## Non-Goals

- No runtime directory scanning during training.
- No motion-terrain recomposition across unrelated pairs.
- No `InstinctMJ` dataset adapter in the first implementation.
- No reset-time terrain rebuilds or terrain geom relocation.
- No lazy motion loading in the first implementation.

## Domain Invariants

Each record in the `Pair dataset` represents one aligned motion-terrain pair.
The motion trajectory and terrain geometry are already expressed in the same
`Pair-local frame`. Runtime code does not infer scale, rotation, or alignment.

The `Tile origin` is the world-space translation of a pair-local frame inside
the terrain tile bank:

```text
terrain_world = terrain_pair_local + tile_origin[pair_id]
motion_world  = motion_pair_local  + tile_origin[pair_id]
```

The reset-time sampling key is the `Pair ID`; motion and terrain are never
sampled independently in this phase.

## Pair Dataset Manifest

The canonical dataset manifest is JSONL. It is created by an offline builder,
not by runtime directory scanning.

Example record:

```json
{"pair_id":"parc/platform/000001","source":"parc","motion_file":"/abs/path/motion.npz","terrain_collision_file":"/abs/path/terrain_collision.json","terrain_visual_file":"/abs/path/multi_boxes.obj","weight":1.0,"category":"platform","motion":{"fps":50,"frames":172,"root_bounds_xy":[[-1.2,-0.8],[2.4,0.9]]},"terrain":{"adapter":"parc_primitive_boxes","box_count":5,"bounds_xy":[[-2.0,-1.5],[3.0,1.5]]}}
```

The builder scans `/home/humanoid/Downloads/Data/parc_initial_aug_g1`, matches
motion files with terrain collision manifests, writes diagnostics, and emits a
JSONL manifest.

Training supports two validation modes:

- `fast`: validate schema, paths, unique pair IDs, and required fields.
- `strict`: recompute motion and terrain diagnostics from source files and
  compare them with manifest values.

Training defaults to `fast`; data generation, debugging, and CI should use
`strict`.

## Terrain Architecture

Terrain source differences are isolated behind `Terrain adapter`s. A terrain
adapter converts one pair dataset record into a common `Terrain tile`.

Initial adapters:

- `ParcPrimitiveBoxAdapter`: reads `terrain_collision.json` and builds primitive
  MuJoCo boxes.
- `OmniRetargetBoxAdapter`: later reads OmniRetarget terrain URDF and box meshes.

The general teacher task consumes only `Terrain tile`s and does not know PARC or
OmniRetarget-specific details.

`PairTerrainBank` builds all selected terrain tiles into the MuJoCo spec at
environment startup. It owns:

- `pair_id -> tile_origin`
- `pair_id -> occupied_bounds_xy`
- deterministic tile packing metadata

It does not use mjlab `TerrainGenerator` as the multi-pair bank abstraction,
because `TerrainGenerator` represents procedural terrain rows/columns rather
than exact `pair_id -> tile` mapping.

## Tile Packing

The first implementation uses deterministic padded grid packing with one global
cell size.

For each pair:

```text
occupied_bounds_xy =
  union(terrain_bounds_xy, motion_root_bounds_xy)
```

Packing uses:

```text
packing_bounds_xy = occupied_bounds_xy inflated by packing_margin
```

Initial default:

```text
packing_margin = 2.0m
```

All tiles remain in their pair-local coordinates and are translated only by
their tile origin. The packer does not recenter or rotate individual pairs.

## Motion Architecture

`MotionLibrary` loads all pair motions at startup onto the training device. It
uses a concat/ragged layout instead of max-length padding.

```text
frame_offsets[pair_index] + local_frame -> global_frame
```

Stored concat tensors include:

- `joint_pos`
- `joint_vel`
- `body_pos_w`
- `body_quat_w`
- `body_lin_vel_w`
- `body_ang_vel_w`

The library also stores frame counts, FPS, pair IDs, and Python-side metadata for
logging and debugging.

## Sampling Architecture

`PairFrameSampler` chooses both the `Pair ID` and the start frame or time bin for
a reset.

Supported modes:

- `independent`: sample `pair_id` by pair weight, then sample a frame bin inside
  that pair.
- `concat_pair_bins`: sample globally over flattened `(pair_id, frame_bin)`.

Default mode:

```text
independent
```

This preserves pair coverage while keeping single-motion internal adaptive
sampling benefits. `concat_pair_bins` is available as an experiment mode for
hard mining.

Adaptive updates use `termination_manager.terminated` as the failure mask. This
excludes timeouts and includes non-timeout tracking failures. The sampler keeps
per-pair frame-bin failure statistics and updates frame-bin weights with an
EMA-style rule similar to the current single-motion adaptive sampling.

## Command Architecture

The general task replaces the single-file `MotionCommand` with
`MultiMotionCommand`.

`MultiMotionCommand` keeps:

- `env_pair_ids`
- `env_time_steps`
- per-env `tile_origin` from `PairTerrainBank`
- references to `MotionLibrary` and `PairFrameSampler`

It exposes the same attributes used by existing observations, rewards, and
terminations: `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`,
`anchor_pos_w`, robot state accessors, and debug metrics.

`scene.env_origins` is not the authority for general teacher pair placement.
The current per-env origin is derived from `env_pair_ids`.

## Task Architecture

Add a new task family:

```text
TT-Tracking-TerrainOracleTeacherGeneral-Unitree-G1
TT-Tracking-TerrainOracleTeacherGeneral-Unitree-G1-Play
```

The new task starts from the current oracle teacher configuration and replaces
only the generalization-specific pieces:

- command: `MultiMotionCommandCfg`
- scene spec: `PairTerrainBank` spec function
- termination: add `out_of_tile_bounds`
- CLI: require `--pair-dataset`

The existing single-pair oracle teacher task remains unchanged.

## Termination

The adaptive failure signal is:

```text
failure_mask = termination_manager.terminated
```

Current tracking terms already populate this with non-timeout failures. The
general task adds `out_of_tile_bounds` as a non-timeout termination.

`out_of_tile_bounds` computes:

```text
root_pair_local_xy = robot_root_world_xy - tile_origin[pair_id].xy
violation = root_pair_local_xy outside occupied_bounds_xy inflated by fail_margin
```

Initial default:

```text
fail_margin = 1.0m
```

This prevents a robot that has run far away from its current pair from reaching
neighboring tiles in the compiled world.

## CLI Shape

Dataset builder:

```bash
python -m terrain_tracking.build_pair_dataset \
  --source parc \
  --root /home/humanoid/Downloads/Data/parc_initial_aug_g1 \
  --output /home/humanoid/Downloads/Data/parc_initial_aug_g1/pair_dataset.jsonl
```

Training:

```bash
python -m terrain_tracking.tasks.general_terrain_tracking.scripts.train \
  --pair-dataset /home/humanoid/Downloads/Data/parc_initial_aug_g1/pair_dataset.jsonl \
  --dataset-validate fast
```

Debug options:

- `--dataset-validate strict`
- `--max-pairs 128`
- `--pair-filter category=stairs`
- `--pair-sampler-mode independent`
- `--pair-sampler-mode concat_pair_bins`

`--max-pairs` and `--pair-filter` are smoke-test and benchmark tools. Full
training defaults to all eligible pairs.

## Validation Plan

Unit tests:

- JSONL parser validates schema, required fields, and unique pair IDs.
- PARC builder matches motion files and terrain collision manifests.
- Strict validation detects stale bounds, frame counts, and box counts.
- Terrain adapters produce stable tile bounds and diagnostics.
- Padded grid packing produces non-overlapping packing bounds.
- MotionLibrary maps `(pair_id, local_frame)` to the expected concat frame.
- PairFrameSampler supports independent and concat-pair-bin modes.
- `out_of_tile_bounds` triggers only outside inflated current-pair bounds.

Smoke tests:

- Build a small PARC manifest subset.
- Compile a small terrain bank and reset/step the general teacher env on CPU.
- Verify observations/rewards/terminations still consume the command interface.
- Run a full PARC terrain-only compile benchmark before long training.

Full run checks:

- Compile full PARC tile bank.
- Log tile count, total primitive boxes, world footprint, and compile time.
- Log sampled pair/category distribution.
- Log adaptive sampling entropy and top-bin probability.
- Log `Episode_Termination/out_of_tile_bounds`.
