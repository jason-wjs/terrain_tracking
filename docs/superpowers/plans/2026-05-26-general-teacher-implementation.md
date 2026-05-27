# General Teacher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first general terrain-aware oracle teacher task over a JSONL pair dataset.

**Architecture:** Add pair dataset parsing/building, terrain tile adapters, a static pair terrain bank, a GPU concat/ragged motion library, a pair-frame sampler, and a `MultiMotionCommand` compatible with existing oracle observations/rewards. Register a new general teacher task without changing the existing single-pair task.

**Tech Stack:** Python 3.10+, mjlab, MuJoCo `MjSpec`, torch, numpy, pytest, ruff.

---

## File Map

- Create `src/terrain_tracking/runtime/pair_dataset.py`: JSONL schema, parser, validation modes.
- Create `src/terrain_tracking/build_pair_dataset.py`: offline PARC manifest builder CLI.
- Create `src/terrain_tracking/scene/terrain_tile.py`: common tile and bounds dataclasses.
- Create `src/terrain_tracking/scene/terrain_adapters/base.py`: terrain adapter protocol.
- Create `src/terrain_tracking/scene/terrain_adapters/parc.py`: PARC primitive box tile adapter.
- Create `src/terrain_tracking/scene/pair_terrain_bank.py`: static tile packing and MuJoCo spec injection.
- Create `src/terrain_tracking/runtime/motion_library.py`: concat/ragged motion storage.
- Create `src/terrain_tracking/runtime/pair_frame_sampler.py`: independent and concat pair-bin reset sampling.
- Create `src/terrain_tracking/tasks/general_terrain_tracking/mdp/multi_motion_command.py`: general command term.
- Create `src/terrain_tracking/tasks/general_terrain_tracking/mdp/terminations.py`: `out_of_tile_bounds`.
- Create `src/terrain_tracking/tasks/general_terrain_tracking/config/g1/env_cfgs.py`: general teacher env cfg.
- Create `src/terrain_tracking/tasks/general_terrain_tracking/config/g1/rl_cfg.py`: general teacher RL cfg.
- Create `src/terrain_tracking/tasks/general_terrain_tracking/config/g1/__init__.py`: task registration.
- Create `src/terrain_tracking/tasks/general_terrain_tracking/scripts/train.py`: train CLI.
- Create `src/terrain_tracking/tasks/general_terrain_tracking/scripts/play.py`: play CLI.
- Modify `src/terrain_tracking/_mjlab_tasks.py`: import general task registration.
- Modify `src/terrain_tracking/scene/__init__.py`: export tile/bank helpers.
- Modify `src/terrain_tracking/runtime/__init__.py`: export dataset/motion/sampler helpers where useful.
- Add focused tests under `tests/`.

## Task 1: Pair Dataset Parser

**Files:**
- Create: `src/terrain_tracking/runtime/pair_dataset.py`
- Test: `tests/test_pair_dataset.py`

- [ ] **Step 1: Write failing parser tests**

Add tests that write a two-line JSONL manifest, load it, and assert:

```python
dataset = PairDataset.load(path, validate="fast")
assert dataset.pair_ids == ("parc/a/000", "parc/b/001")
assert dataset.records[0].source == "parc"
assert dataset.records[0].motion.frames == 4
assert dataset.records[0].terrain.adapter == "parc_primitive_boxes"
```

Also add duplicate `pair_id` and missing file tests.

- [ ] **Step 2: Verify red**

Run:

```bash
uv run pytest -q tests/test_pair_dataset.py
```

Expected: import failure for `terrain_tracking.runtime.pair_dataset`.

- [ ] **Step 3: Implement parser**

Implement frozen dataclasses:

```python
PairDatasetRecord
MotionDiagnostics
TerrainDiagnostics
PairDataset
```

Implement `PairDataset.load(path, validate="fast")`. `fast` validates schema,
required fields, path existence, and unique pair IDs. `strict` initially calls
the same checks and is extended in later tasks.

- [ ] **Step 4: Verify green**

Run:

```bash
uv run pytest -q tests/test_pair_dataset.py
```

Expected: pass.

## Task 2: PARC Dataset Builder

**Files:**
- Create: `src/terrain_tracking/build_pair_dataset.py`
- Test: `tests/test_build_pair_dataset.py`

- [ ] **Step 1: Write failing builder tests**

Use `tests.helpers.create_motion_clip()` and
`tests.helpers.create_heightfield_collision_manifest()` to create a synthetic
PARC-like tree with two records. Assert the builder writes JSONL with stable
pair IDs, `category`, `motion.frames`, `motion.root_bounds_xy`, terrain adapter,
box count, and bounds.

- [ ] **Step 2: Verify red**

Run:

```bash
uv run pytest -q tests/test_build_pair_dataset.py
```

Expected: import failure for `terrain_tracking.build_pair_dataset`.

- [ ] **Step 3: Implement builder**

Implement:

```python
BuildPairDatasetConfig
build_parc_pair_dataset(config) -> PairDataset
entry_point()
```

The PARC scanner finds `motion.npz` and matching `terrain_collision.json` under
the root, skips unmatched terrain files, and writes one JSON object per line.

- [ ] **Step 4: Verify green**

Run:

```bash
uv run pytest -q tests/test_build_pair_dataset.py tests/test_pair_dataset.py
```

Expected: pass.

## Task 3: Terrain Tile and PARC Adapter

**Files:**
- Create: `src/terrain_tracking/scene/terrain_tile.py`
- Create: `src/terrain_tracking/scene/terrain_adapters/__init__.py`
- Create: `src/terrain_tracking/scene/terrain_adapters/base.py`
- Create: `src/terrain_tracking/scene/terrain_adapters/parc.py`
- Modify: `src/terrain_tracking/scene/__init__.py`
- Test: `tests/test_terrain_tile_adapters.py`

- [ ] **Step 1: Write failing adapter tests**

Build a `PairDatasetRecord` pointing to a heightfield collision manifest. Assert:

```python
tile = ParcPrimitiveBoxAdapter().build_tile(record)
assert tile.pair_id == record.pair_id
assert tile.bounds_xy.shape == (2, 2)
assert len(tile.boxes) == record.terrain.box_count
```

- [ ] **Step 2: Verify red**

Run:

```bash
uv run pytest -q tests/test_terrain_tile_adapters.py
```

Expected: import failure for adapter modules.

- [ ] **Step 3: Implement adapter**

Wrap existing `build_primitive_box_tile()` output in a `TerrainTile` dataclass
containing `pair_id`, `boxes`, `terrain_bounds_xy`, `motion_root_bounds_xy`, and
`occupied_bounds_xy`.

- [ ] **Step 4: Verify green**

Run:

```bash
uv run pytest -q tests/test_terrain_tile_adapters.py tests/test_primitive_box_terrain.py
```

Expected: pass.

## Task 4: Pair Terrain Bank

**Files:**
- Create: `src/terrain_tracking/scene/pair_terrain_bank.py`
- Test: `tests/test_pair_terrain_bank.py`

- [ ] **Step 1: Write failing bank tests**

Construct two `TerrainTile`s with known bounds and one box each. Assert:

```python
bank = PairTerrainBank.build(tiles, packing_margin=2.0)
assert bank.tile_origins.shape == (2, 3)
assert bank.pair_index_by_id["pair_a"] == 0
assert bank.packing_cell_size[0] >= expected_width
```

Call `bank.add_to_spec(mujoco.MjSpec())`, compile, and assert geom names include
both pair tiles.

- [ ] **Step 2: Verify red**

Run:

```bash
uv run pytest -q tests/test_pair_terrain_bank.py
```

Expected: import failure.

- [ ] **Step 3: Implement bank**

Implement deterministic padded grid packing. Add boxes as static world geoms
with names containing sanitized pair IDs and box names. Store `tile_origins`,
`occupied_bounds_xy`, and `pair_index_by_id`.

- [ ] **Step 4: Verify green**

Run:

```bash
uv run pytest -q tests/test_pair_terrain_bank.py
```

Expected: pass.

## Task 5: Motion Library

**Files:**
- Create: `src/terrain_tracking/runtime/motion_library.py`
- Test: `tests/test_motion_library.py`

- [ ] **Step 1: Write failing motion library tests**

Create two `motion.npz` clips with different frame counts. Assert:

```python
lib = MotionLibrary.from_records(records, device="cpu")
idx = lib.global_frame_index(pair_indices=torch.tensor([1]), local_frames=torch.tensor([2]))
assert int(idx[0]) == lib.frame_offsets[1] + 2
assert lib.joint_pos(idx).shape[0] == 1
```

- [ ] **Step 2: Verify red**

Run:

```bash
uv run pytest -q tests/test_motion_library.py
```

Expected: import failure.

- [ ] **Step 3: Implement concat/ragged loader**

Load all required mjlab motion arrays, concatenate along frame dimension, and
store `frame_offsets`, `frame_counts`, `pair_ids`, and `pair_index_by_id`.

- [ ] **Step 4: Verify green**

Run:

```bash
uv run pytest -q tests/test_motion_library.py
```

Expected: pass.

## Task 6: Pair Frame Sampler

**Files:**
- Create: `src/terrain_tracking/runtime/pair_frame_sampler.py`
- Test: `tests/test_pair_frame_sampler.py`

- [ ] **Step 1: Write failing sampler tests**

Use three pairs with frame counts `[10, 20, 30]`. Assert independent mode returns
valid pair indices and local frames. Update with a failure mask and assert
adaptive weights remain normalized per pair. Assert concat mode returns valid
pairs and frames from global bins.

- [ ] **Step 2: Verify red**

Run:

```bash
uv run pytest -q tests/test_pair_frame_sampler.py
```

Expected: import failure.

- [ ] **Step 3: Implement sampler**

Implement `PairFrameSamplerCfg`, `PairFrameSampler`, `sample(env_ids)`, and
`update_failures(pair_indices, local_frames, failure_mask)`. Support
`mode="independent"` and `mode="concat_pair_bins"`.

- [ ] **Step 4: Verify green**

Run:

```bash
uv run pytest -q tests/test_pair_frame_sampler.py
```

Expected: pass.

## Task 7: MultiMotionCommand and out_of_tile_bounds

**Files:**
- Create: `src/terrain_tracking/tasks/general_terrain_tracking/mdp/__init__.py`
- Create: `src/terrain_tracking/tasks/general_terrain_tracking/mdp/multi_motion_command.py`
- Create: `src/terrain_tracking/tasks/general_terrain_tracking/mdp/terminations.py`
- Test: `tests/test_multi_motion_command_unit.py`
- Test: `tests/test_out_of_tile_bounds.py`

- [ ] **Step 1: Write failing unit tests**

Test `out_of_tile_bounds` with a lightweight fake env/command object. Test that
the command config type can be constructed and exposes expected config fields.

- [ ] **Step 2: Verify red**

Run:

```bash
uv run pytest -q tests/test_multi_motion_command_unit.py tests/test_out_of_tile_bounds.py
```

Expected: import failure.

- [ ] **Step 3: Implement minimal command and termination**

Implement `MultiMotionCommandCfg` and `MultiMotionCommand` by following
mjlab `MotionCommand` public attributes. Use `MotionLibrary`,
`PairFrameSampler`, and `PairTerrainBank` references from config. Implement
root reset, time-step updates, body/anchor properties, adaptive sampler update,
and `out_of_tile_bounds`.

- [ ] **Step 4: Verify green**

Run:

```bash
uv run pytest -q tests/test_multi_motion_command_unit.py tests/test_out_of_tile_bounds.py
```

Expected: pass.

## Task 8: General Teacher Task Registration

**Files:**
- Create: `src/terrain_tracking/tasks/general_terrain_tracking/__init__.py`
- Create: `src/terrain_tracking/tasks/general_terrain_tracking/config/__init__.py`
- Create: `src/terrain_tracking/tasks/general_terrain_tracking/config/g1/__init__.py`
- Create: `src/terrain_tracking/tasks/general_terrain_tracking/config/g1/env_cfgs.py`
- Create: `src/terrain_tracking/tasks/general_terrain_tracking/config/g1/rl_cfg.py`
- Modify: `src/terrain_tracking/_mjlab_tasks.py`
- Modify: `tests/test_registry.py`
- Test: `tests/test_general_terrain_tracking_config.py`

- [ ] **Step 1: Write failing config/registry tests**

Assert the general task registers:

```python
assert "TT-Tracking-TerrainOracleTeacherGeneral-Unitree-G1" in tasks
```

Assert env cfg contains `MultiMotionCommandCfg`, keeps oracle teacher terrain
scan, and includes `out_of_tile_bounds`.

- [ ] **Step 2: Verify red**

Run:

```bash
uv run pytest -q tests/test_general_terrain_tracking_config.py tests/test_registry.py
```

Expected: general imports fail or task missing.

- [ ] **Step 3: Implement task cfg and registration**

Reuse `unitree_g1_oracle_teacher_terrain_tracking_env_cfg()`, replace command
cfg with `MultiMotionCommandCfg`, add general termination, and register train
and play task IDs.

- [ ] **Step 4: Verify green**

Run:

```bash
uv run pytest -q tests/test_general_terrain_tracking_config.py tests/test_registry.py
```

Expected: pass.

## Task 9: General Train/Play CLIs and Smoke Test

**Files:**
- Create: `src/terrain_tracking/tasks/general_terrain_tracking/scripts/__init__.py`
- Create: `src/terrain_tracking/tasks/general_terrain_tracking/scripts/train.py`
- Create: `src/terrain_tracking/tasks/general_terrain_tracking/scripts/play.py`
- Test: `tests/test_general_env_smoke.py`
- Modify: `tests/test_cli_help.py`

- [ ] **Step 1: Write failing CLI and smoke tests**

Assert `--help` works for the new train/play modules. Build a two-pair JSONL
manifest, create a CPU env with one env, call reset and one zero-action step.

- [ ] **Step 2: Verify red**

Run:

```bash
uv run pytest -q tests/test_general_env_smoke.py tests/test_cli_help.py
```

Expected: imports fail or CLI modules missing.

- [ ] **Step 3: Implement scripts**

Implement dataset loading, strict/fast validation option, `--max-pairs`,
`--pair-filter`, `--pair-sampler-mode`, and task launch parity with existing
single-pair train/play scripts.

- [ ] **Step 4: Verify green**

Run:

```bash
uv run pytest -q tests/test_general_env_smoke.py tests/test_cli_help.py
```

Expected: pass.

## Task 10: Full Verification

**Files:**
- Modify as needed from earlier tasks.

- [ ] **Step 1: Run focused test set**

Run:

```bash
uv run pytest -q tests/test_pair_dataset.py tests/test_build_pair_dataset.py tests/test_terrain_tile_adapters.py tests/test_pair_terrain_bank.py tests/test_motion_library.py tests/test_pair_frame_sampler.py tests/test_general_terrain_tracking_config.py tests/test_general_env_smoke.py
```

Expected: pass.

- [ ] **Step 2: Run existing regression tests most likely affected**

Run:

```bash
uv run pytest -q tests/test_env_smoke.py tests/test_registry.py tests/test_oracle_terrain_tracking_config.py tests/test_primitive_box_terrain.py
```

Expected: pass.

- [ ] **Step 3: Run lint**

Run:

```bash
uv run ruff check src tests
```

Expected: pass.

- [ ] **Step 4: Commit**

Commit implementation files only. Leave unrelated pre-existing files unstaged.

```bash
git status --short
git add src tests docs/superpowers/plans/2026-05-26-general-teacher-implementation.md
git commit -m "feat: add general terrain teacher task"
```
