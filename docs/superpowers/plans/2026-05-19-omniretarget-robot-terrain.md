# OmniRetarget Robot-Terrain Single Pair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support importing one OmniRetarget `robot-terrain` sample into `terrain_tracking` and running the existing terrain tracking task against it.

**Architecture:** Add a focused converter for OmniRetarget `qpos` clips and a focused scene backend for OmniRetarget URDF box terrains. The existing pair manifest remains the user-facing runtime contract; `--collision-backend omniretarget_boxes` selects the new terrain path while preserving existing PARC backends.

**Tech Stack:** Python 3.13, NumPy, MuJoCo Python bindings, mjlab G1 MJCF, pytest, existing `terrain_tracking` pair manifest and training scripts.

---

## File Map

- Create `src/terrain_tracking/omniretarget_motion.py`
  - Parse OmniRetarget `qpos` clips.
  - Resample from source fps to 50 Hz.
  - Use mjlab G1 MJCF for FK.
  - Write mjlab tracking motion `.npz`.
- Create `src/terrain_tracking/scene/omniretarget_boxes.py`
  - Parse OmniRetarget terrain URDF collision meshes.
  - Load scaled OBJ vertices.
  - Fit each mesh as a yawed MuJoCo box primitive.
  - Add obstacle boxes and a ground box to `MjSpec`.
- Create `src/terrain_tracking/convert_omniretarget_robot_terrain.py`
  - Single-sample CLI that writes `motion.npz`, `pair.json`, and `meta.json`.
- Modify `src/terrain_tracking/runtime/apply_pair.py`
  - Accept `collision_backend="omniretarget_boxes"`.
  - Chain the existing scene spec function with the OmniRetarget terrain spec.
- Modify `tests/helpers.py`
  - Add helpers for tiny OmniRetarget qpos clips and terrain URDF/OBJ fixtures.
- Create `tests/test_omniretarget_motion.py`
- Create `tests/test_omniretarget_boxes.py`
- Create `tests/test_convert_omniretarget_robot_terrain.py`
- Modify `tests/test_env_smoke.py`
  - Add one small build/reset/step smoke for the new backend.
- Modify `tests/test_cli_help.py`
  - Confirm the new converter module exposes CLI help.
- Add docs in `README.md`
  - Show the single-sample OmniRetarget conversion and training commands.

## Task 1: Motion Conversion Core

**Files:**
- Create: `src/terrain_tracking/omniretarget_motion.py`
- Test: `tests/test_omniretarget_motion.py`
- Modify: `tests/helpers.py`

- [ ] **Step 1: Write tests for qpos parsing and resampling**

Add tests that create a synthetic OmniRetarget clip with `qpos.shape == (4, 36)` and `fps == 30`. Verify:

- root quaternion and position are parsed from columns `0:7`
- joints are parsed from columns `7:36`
- resampling to 50 Hz produces monotonically increasing output times
- invalid qpos shapes raise `ValueError`

Run:

```bash
uv run pytest -q tests/test_omniretarget_motion.py
```

Expected: fail because `terrain_tracking.omniretarget_motion` does not exist.

- [ ] **Step 2: Implement qpos parsing and linear resampling**

Implement:

- `OmniRetargetClip`
- `load_omniretarget_clip(path: str | Path) -> OmniRetargetClip`
- `resample_clip(clip: OmniRetargetClip, output_fps: int = 50) -> OmniRetargetClip`

Use normalized linear interpolation for root position and joints. Normalize interpolated quaternions after interpolation. Keep this first implementation deterministic and explicit.

- [ ] **Step 3: Run tests**

Run:

```bash
uv run pytest -q tests/test_omniretarget_motion.py
```

Expected: qpos parsing and resampling tests pass.

- [ ] **Step 4: Add FK motion emission tests**

Extend `tests/test_omniretarget_motion.py` to call `emit_mjlab_motion_npz(...)` on the synthetic clip. Verify the written file contains:

- `fps`
- `joint_pos`
- `joint_vel`
- `body_pos_w`
- `body_quat_w`
- `body_lin_vel_w`
- `body_ang_vel_w`

Verify `joint_pos.shape[1] == 29`, `body_pos_w.shape[-1] == 3`, and `body_quat_w.shape[-1] == 4`.

- [ ] **Step 5: Implement FK motion emission**

Use the current mjlab Unitree G1 XML:

```python
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
```

Load it with `mujoco.MjModel.from_xml_path`, allocate `MjData`, fill `data.qpos` as `[x, y, z, qw, qx, qy, qz, joints...]`, run `mujoco.mj_forward`, and collect all body poses. Use finite differences over output frame time for velocities.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
uv run pytest -q tests/test_omniretarget_motion.py
```

Commit:

```bash
git add src/terrain_tracking/omniretarget_motion.py tests/test_omniretarget_motion.py tests/helpers.py
git commit -m "feat: convert omniretarget qpos motion clips"
```

## Task 2: OmniRetarget Terrain Box Parsing

**Files:**
- Create: `src/terrain_tracking/scene/omniretarget_boxes.py`
- Test: `tests/test_omniretarget_boxes.py`
- Modify: `tests/helpers.py`

- [ ] **Step 1: Write terrain fixture helpers**

Add helper functions that write:

- a rectangular box OBJ with 8 vertices and 12 triangular faces
- a URDF with a collision mesh referencing that OBJ and a non-uniform scale

- [ ] **Step 2: Write failing parser tests**

Create tests that verify:

- `load_omniretarget_terrain_boxes(urdf_path)` resolves mesh paths relative to the URDF
- duplicate visual/collision mesh entries do not double-count the same collision
- mesh scale is applied
- fitted box half-size and center match the fixture

Run:

```bash
uv run pytest -q tests/test_omniretarget_boxes.py
```

Expected: fail because the module does not exist.

- [ ] **Step 3: Implement URDF and OBJ parsing**

Use `xml.etree.ElementTree` for URDF. Parse collision meshes only. Use a small OBJ vertex parser that reads `v x y z` lines; face parsing is not required for box fitting.

Define:

- `OmniRetargetBox`
- `load_omniretarget_terrain_boxes(path: str | Path) -> tuple[OmniRetargetBox, ...]`

- [ ] **Step 4: Implement yawed box fitting**

Fit the box by PCA in the XY plane and min/max extents in the rotated frame. Preserve yaw. Validate that all scaled vertices lie on the fitted box surface within a tolerance.

- [ ] **Step 5: Add spec compilation test**

Add a test that calls `make_omniretarget_boxes_spec_fn(...)` on an empty `MjSpec`, compiles it, and confirms geoms are MuJoCo boxes. Include a ground/base box.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
uv run pytest -q tests/test_omniretarget_boxes.py
```

Commit:

```bash
git add src/terrain_tracking/scene/omniretarget_boxes.py tests/test_omniretarget_boxes.py tests/helpers.py
git commit -m "feat: load omniretarget terrain boxes"
```

## Task 3: Runtime Backend Integration

**Files:**
- Modify: `src/terrain_tracking/runtime/apply_pair.py`
- Test: `tests/test_env_smoke.py`

- [ ] **Step 1: Add failing backend validation test**

Add a test that builds a pair manifest with `terrain_file` pointing to an OmniRetarget terrain URDF and calls:

```python
apply_pair_manifest_to_env_cfg(
  cfg,
  pair_json,
  collision_backend="omniretarget_boxes",
)
```

Verify `cfg.scene.spec_fn` is set and no `terrain_collision_file` is required.

- [ ] **Step 2: Implement backend registration**

Modify `apply_pair_manifest_to_env_cfg`:

- accept `"omniretarget_boxes"` in the backend allowlist
- for this backend, chain `make_omniretarget_boxes_spec_fn(pair.terrain_file)` into `cfg.scene.spec_fn`
- set contact buffers high enough for obstacle boxes plus ground
- preserve existing `primitive_boxes`, `hfield`, and `mesh` behavior

- [ ] **Step 3: Add env smoke**

Add a `num_envs=1` smoke that uses a tiny synthetic OmniRetarget converted motion and terrain URDF, builds the env, resets, steps once with zero action, and closes the env.

- [ ] **Step 4: Run backend tests and commit**

Run:

```bash
uv run pytest -q tests/test_env_smoke.py -k omniretarget
```

Commit:

```bash
git add src/terrain_tracking/runtime/apply_pair.py tests/test_env_smoke.py
git commit -m "feat: add omniretarget terrain backend"
```

## Task 4: Single-Sample Converter CLI

**Files:**
- Create: `src/terrain_tracking/convert_omniretarget_robot_terrain.py`
- Test: `tests/test_convert_omniretarget_robot_terrain.py`
- Modify: `tests/test_cli_help.py`

- [ ] **Step 1: Write converter tests**

Test:

- `resolve_terrain_urdf("climb_00_z_scale_1.0", terrain_root)` returns `terrain_root/climb_00/multi_boxes_z_scale_1.0.urdf`
- conversion writes `motion.npz`, `pair.json`, and `meta.json`
- `pair.json["terrain_file"]` points to the URDF
- `meta.json` records source fps, output fps, source frame count, output frame count, and terrain file

- [ ] **Step 2: Implement CLI and conversion function**

Implement:

- `ConvertOmniRetargetRobotTerrainConfig`
- `resolve_terrain_urdf(...)`
- `convert_omniretarget_robot_terrain(...)`
- `main()`

Use the existing `write_pair_manifest_bundle(...)` to write `pair.json` and `meta.json`, then update `meta.json` with OmniRetarget-specific fields.

- [ ] **Step 3: Add CLI help test**

Extend `tests/test_cli_help.py` to run:

```bash
uv run python -m terrain_tracking.convert_omniretarget_robot_terrain --help
```

Expected: exit code 0 and help text includes `--motion-file`.

- [ ] **Step 4: Run focused tests and commit**

Run:

```bash
uv run pytest -q tests/test_convert_omniretarget_robot_terrain.py tests/test_cli_help.py
```

Commit:

```bash
git add src/terrain_tracking/convert_omniretarget_robot_terrain.py tests/test_convert_omniretarget_robot_terrain.py tests/test_cli_help.py
git commit -m "feat: add omniretarget robot-terrain converter"
```

## Task 5: Documentation And Real-Sample Smoke

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document single-sample workflow**

Add a README section with:

- expected OmniRetarget input files
- conversion command
- play command with `--collision-backend omniretarget_boxes`
- training smoke command with `--env.scene.num-envs 1` and tiny iteration count

- [ ] **Step 2: Run the full unit suite for changed areas**

Run:

```bash
uv run pytest -q tests/test_omniretarget_motion.py tests/test_omniretarget_boxes.py tests/test_convert_omniretarget_robot_terrain.py tests/test_env_smoke.py -k "omniretarget or not requires_real_data"
```

- [ ] **Step 3: Convert one real local sample**

Run:

```bash
uv run python -m terrain_tracking.convert_omniretarget_robot_terrain \
  --motion-file /path/to/OmniRetarget_Dataset/robot-terrain/climb_00_z_scale_1.0.npz \
  --terrain-root /path/to/OmniRetarget_Dataset/models/terrain \
  --output-dir /tmp/tt_converted_omniretarget \
  --sample-name climb_00_z_scale_1.0
```

Expected:

```text
/tmp/tt_converted_omniretarget/climb_00_z_scale_1.0/pair.json
```

- [ ] **Step 4: Build/reset/step real sample**

Run a small env smoke against the generated pair:

```bash
uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.play \
  --pair-manifest /tmp/tt_converted_omniretarget/climb_00_z_scale_1.0/pair.json \
  --collision-backend omniretarget_boxes \
  --task TT-Tracking-TerrainOracleHeight-Unitree-G1 \
  --agent zero \
  --num-envs 1 \
  --headless
```

If `play` does not expose `--headless`, use the existing env smoke path instead of opening a viewer.

- [ ] **Step 5: Run short training smoke**

Run:

```bash
uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.train \
  --pair-manifest /tmp/tt_converted_omniretarget/climb_00_z_scale_1.0/pair.json \
  --collision-backend omniretarget_boxes \
  --task TT-Tracking-TerrainOracleHeight-Unitree-G1 \
  --env.scene.num-envs 1 \
  --agent.max-iterations 1 \
  --agent.experiment-name tt_omniretarget_smoke \
  --agent.run-name climb_00_z_scale_1_0_smoke
```

- [ ] **Step 6: Commit docs and final verification**

Run:

```bash
uv run pytest -q tests/test_omniretarget_motion.py tests/test_omniretarget_boxes.py tests/test_convert_omniretarget_robot_terrain.py tests/test_env_smoke.py -k omniretarget
```

Commit:

```bash
git add README.md
git commit -m "docs: document omniretarget robot-terrain workflow"
```

Push:

```bash
git push
```
