# TT Convert Pair Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a conversion path inside `terrain_tracking/` that turns one retargeted PARC-derived `qpos` motion plus one terrain mesh into one self-contained pair bundle consumable by the blind terrain tracking task.

**Architecture:** Keep task-facing entrypoints under `tasks/blind_terrain_tracking/scripts/`, but place real conversion logic in `src/terrain_tracking/data_conversion/`. Parse the retargeted clip, replay it through official installed `mjlab` G1 assets to emit canonical tracking arrays, then package `motion.npz`, `terrain.obj`, and `pair.json` into one output directory.

**Tech Stack:** Python, `mjlab[cu128]==1.3.0`, MuJoCo via installed `mjlab`, `numpy`, `json`, `pytest`, `uv`.

---

## Scope Lock

- Project root: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking`
- Runtime dependency source: installed official `mjlab`, not `controller/mjlab`
- CLI entrypoint:
  `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/tasks/blind_terrain_tracking/scripts/convert.py`
- Shell wrapper:
  `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/scripts/convert_pair.sh`
- Core modules:
  - `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/data_conversion/retarget_input.py`
  - `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/data_conversion/mjlab_motion.py`
  - `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/data_conversion/pair_bundle.py`
  - `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/data_conversion/convert_pair.py`
- Phase 1 validation samples:
  - `/tmp/parc_process_workspace/retargeted/platform_001_original.npz`
  - `/tmp/parc_process_workspace/workspace/platform_001/multi_boxes.obj`
  - `/tmp/parc_process_workspace/retargeted/mid_blocks_004_dm_original.npz`
  - `/tmp/parc_process_workspace/workspace/mid_blocks_004_dm/multi_boxes.obj`

## Task 1: Scaffold Conversion Package and CLI Entry

**Files:**
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/data_conversion/__init__.py`
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/data_conversion/convert_pair.py`
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/tasks/blind_terrain_tracking/scripts/convert.py`
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/scripts/convert_pair.sh`
- Test: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/tests/test_cli_help.py`

**Step 1: Write the failing CLI help test**

Write a test that runs:

```python
result = subprocess.run(
    [
        sys.executable,
        "-m",
        "terrain_tracking.tasks.blind_terrain_tracking.scripts.convert",
        "--help",
    ],
    capture_output=True,
    text=True,
)
assert result.returncode == 0
assert "--retarget-npz" in result.stdout
assert "--terrain-obj" in result.stdout
assert "--output-dir" in result.stdout
```

**Step 2: Run test to verify it fails**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/test_cli_help.py
```

Expected: FAIL because the convert entrypoint does not exist.

**Step 3: Write minimal implementation**

Create:

- `data_conversion/__init__.py`
- `data_conversion/convert_pair.py` with a placeholder callable
- `tasks/blind_terrain_tracking/scripts/convert.py` with `argparse` and `--help`
- `scripts/convert_pair.sh` that sources repo root and runs the module

Minimal CLI skeleton:

```python
parser.add_argument("--retarget-npz", required=True)
parser.add_argument("--terrain-obj", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--output-fps", type=int, default=50)
```

**Step 4: Run test to verify it passes**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/test_cli_help.py
```

Expected: PASS

**Step 5: Commit**

```bash
git add \
  src/terrain_tracking/data_conversion/__init__.py \
  src/terrain_tracking/data_conversion/convert_pair.py \
  src/terrain_tracking/tasks/blind_terrain_tracking/scripts/convert.py \
  scripts/convert_pair.sh \
  tests/test_cli_help.py
git commit -m "feat: scaffold terrain tracking pair conversion entrypoints"
```

## Task 2: Parse Retargeted QPOS Input

**Files:**
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/data_conversion/retarget_input.py`
- Test: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/tests/test_data_conversion_retarget_input.py`

**Step 1: Write the failing parser tests**

Cover:

- valid `.npz` with `qpos` and `fps` loads
- missing `qpos` fails clearly
- malformed shape fails clearly
- root and joint slices are split correctly

Use a tiny synthetic file:

```python
qpos = np.zeros((4, 36), dtype=np.float32)
qpos[:, 0] = [0.0, 0.1, 0.2, 0.3]
qpos[:, 3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
```

Assert:

```python
clip.root_pos.shape == (4, 3)
clip.root_quat_xyzw.shape == (4, 4)
clip.joint_pos.shape == (4, 29)
clip.fps == 30
```

**Step 2: Run test to verify it fails**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/test_data_conversion_retarget_input.py
```

Expected: FAIL because the parser module does not exist.

**Step 3: Write minimal implementation**

Implement:

- frozen dataclass `RetargetInputClip`
- loader `load_retarget_input(path)`
- validation:
  - `qpos` exists
  - rank is 2
  - columns are at least `7 + 29`
- slicing:

```python
root_pos = qpos[:, 0:3]
root_quat_xyzw = qpos[:, 3:7]
joint_pos = qpos[:, 7:36]
fps = int(np.asarray(data["fps"]).item())
```

**Step 4: Run test to verify it passes**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/test_data_conversion_retarget_input.py
```

Expected: PASS

**Step 5: Commit**

```bash
git add \
  src/terrain_tracking/data_conversion/retarget_input.py \
  tests/test_data_conversion_retarget_input.py
git commit -m "feat: parse retargeted qpos inputs for pair conversion"
```

## Task 3: Write Pair Bundle Packaging

**Files:**
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/data_conversion/pair_bundle.py`
- Test: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/tests/test_data_conversion_pair_bundle.py`

**Step 1: Write the failing bundle tests**

Cover:

- output directory is created
- `terrain.obj` is copied
- `pair.json` uses relative paths
- default transform fields are identity
- `meta.json` includes source trace fields

Assert:

```python
payload = json.loads((bundle_dir / "pair.json").read_text())
assert payload["motion_file"] == "motion.npz"
assert payload["terrain_file"] == "terrain.obj"
assert payload["terrain_scale"] == [1.0, 1.0, 1.0]
```

**Step 2: Run test to verify it fails**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/test_data_conversion_pair_bundle.py
```

Expected: FAIL because the bundle writer does not exist.

**Step 3: Write minimal implementation**

Implement:

- helper to create `<output_root>/<sample_name>/`
- helper to write `motion.npz`
- helper to copy `terrain.obj`
- helper to write `pair.json`
- helper to write `meta.json`

Minimal `pair.json` payload:

```python
payload = {
    "motion_file": "motion.npz",
    "terrain_file": "terrain.obj",
    "terrain_translation": [0.0, 0.0, 0.0],
    "terrain_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    "terrain_scale": [1.0, 1.0, 1.0],
}
```

**Step 4: Run test to verify it passes**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/test_data_conversion_pair_bundle.py
```

Expected: PASS

**Step 5: Commit**

```bash
git add \
  src/terrain_tracking/data_conversion/pair_bundle.py \
  tests/test_data_conversion_pair_bundle.py
git commit -m "feat: write self-contained terrain tracking pair bundles"
```

## Task 4: Emit Canonical MJLAB Motion Arrays

**Files:**
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/data_conversion/mjlab_motion.py`
- Test: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/tests/test_data_conversion_convert_pair.py`

**Step 1: Write the failing canonicalization test**

Write a small test around one tiny synthetic clip that verifies:

- emitted `joint_pos` has shape `(T, 29)`
- emitted `joint_vel` has shape `(T, 29)`
- emitted `body_pos_w[..., 3] == 3`
- emitted `body_quat_w[..., 4] == 4`
- the output includes `fps`

Assert at minimum:

```python
assert motion["joint_pos"].ndim == 2
assert motion["joint_vel"].shape == motion["joint_pos"].shape
assert motion["body_pos_w"].shape[0] == motion["joint_pos"].shape[0]
assert motion["body_quat_w"].shape[-1] == 4
```

**Step 2: Run test to verify it fails**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/test_data_conversion_convert_pair.py::test_emit_mjlab_motion_arrays
```

Expected: FAIL because the canonical emitter does not exist.

**Step 3: Write minimal implementation**

Implement:

- a function that accepts parsed retarget input and `output_fps`
- optional resampling to target fps
- `mjlab` scene setup using installed package APIs
- per-frame replay:

```python
root_states[:, 0:3] = root_pos
root_states[:, 3:7] = root_quat_xyzw
joint_pos[:, robot_joint_indexes] = joint_frame
robot.write_root_state_to_sim(root_states)
robot.write_joint_state_to_sim(joint_pos, joint_vel)
sim.forward()
scene.update(sim.mj_model.opt.timestep)
```

- logging of:
  - `fps`
  - `joint_pos`
  - `joint_vel`
  - `body_pos_w`
  - `body_quat_w`
  - `body_lin_vel_w`
  - `body_ang_vel_w`

**Step 4: Run test to verify it passes**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/test_data_conversion_convert_pair.py::test_emit_mjlab_motion_arrays
```

Expected: PASS

**Step 5: Commit**

```bash
git add \
  src/terrain_tracking/data_conversion/mjlab_motion.py \
  tests/test_data_conversion_convert_pair.py
git commit -m "feat: emit canonical mjlab motion arrays for pair conversion"
```

## Task 5: Wire High-Level Conversion Orchestration

**Files:**
- Modify: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/data_conversion/convert_pair.py`
- Modify: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/tasks/blind_terrain_tracking/scripts/convert.py`
- Test: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/tests/test_data_conversion_convert_pair.py`

**Step 1: Write the failing end-to-end conversion test**

Write a test that:

- creates a tiny synthetic retarget input
- creates a tiny terrain `.obj`
- runs the high-level conversion API
- asserts bundle files exist
- loads the resulting `pair.json` through the runtime parser

Assert:

```python
pair = PairManifest.load(bundle_dir / "pair.json")
assert pair.motion_file.exists()
assert pair.terrain_file.exists()
```

**Step 2: Run test to verify it fails**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/test_data_conversion_convert_pair.py::test_convert_pair_end_to_end
```

Expected: FAIL because the orchestrator only contains placeholders.

**Step 3: Write minimal implementation**

Implement:

- high-level `convert_pair(...)`
- CLI wiring that forwards:
  - `--retarget-npz`
  - `--terrain-obj`
  - `--output-dir`
  - `--sample-name`
  - `--output-fps`
- return the final bundle directory path

Minimal orchestration:

```python
clip = load_retarget_input(retarget_npz)
motion = emit_mjlab_motion(clip, output_fps=output_fps)
bundle = write_pair_bundle(...)
return bundle
```

**Step 4: Run test to verify it passes**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/test_data_conversion_convert_pair.py
```

Expected: PASS

**Step 5: Commit**

```bash
git add \
  src/terrain_tracking/data_conversion/convert_pair.py \
  src/terrain_tracking/tasks/blind_terrain_tracking/scripts/convert.py \
  tests/test_data_conversion_convert_pair.py
git commit -m "feat: wire end-to-end pair conversion workflow"
```

## Task 6: Validate the Generated Bundle Against Runtime

**Files:**
- Modify: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/tests/test_env_smoke.py`

**Step 1: Write the failing runtime smoke test**

Add a test that:

- generates a tiny converted pair bundle
- loads env config via `build_paired_env(...)`
- asserts the motion file is attached
- asserts the scene spec callback is populated

Assert:

```python
env, _ = build_paired_env(...)
motion_cfg = env.cfg.commands["motion"]
assert motion_cfg.motion_file.endswith("motion.npz")
assert env.cfg.scene.spec_fn is not None
```

**Step 2: Run test to verify it fails**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/test_env_smoke.py::test_converted_pair_bundle_loads
```

Expected: FAIL because the new converter is not yet integrated into the test helpers.

**Step 3: Write minimal implementation**

Update test helpers or smoke setup to build a minimal converted pair before env creation.

Keep the runtime untouched unless test setup truly requires a small helper.

**Step 4: Run test to verify it passes**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/test_env_smoke.py::test_converted_pair_bundle_loads
```

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_env_smoke.py
git commit -m "test: verify converted pair bundles load in terrain tracking env"
```

## Task 7: Manual Validation on Real PARC-Derived Samples

**Files:**
- No code changes required unless a real mismatch is discovered

**Step 1: Convert `platform_001`**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
bash scripts/convert_pair.sh \
  --retarget-npz /tmp/parc_process_workspace/retargeted/platform_001_original.npz \
  --terrain-obj /tmp/parc_process_workspace/workspace/platform_001/multi_boxes.obj \
  --output-dir /tmp/tt_converted/platform_001
```

Expected: bundle directory contains `motion.npz`, `terrain.obj`, `pair.json`, `meta.json`.

**Step 2: Play `platform_001`**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
bash scripts/play.sh \
  --pair-manifest /tmp/tt_converted/platform_001/pair.json \
  --agent zero
```

Expected: robot and terrain both load, motion command initializes, no format/runtime error is raised at startup.

**Step 3: Convert `mid_blocks_004_dm`**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
bash scripts/convert_pair.sh \
  --retarget-npz /tmp/parc_process_workspace/retargeted/mid_blocks_004_dm_original.npz \
  --terrain-obj /tmp/parc_process_workspace/workspace/mid_blocks_004_dm/multi_boxes.obj \
  --output-dir /tmp/tt_converted/mid_blocks_004_dm
```

Expected: bundle directory contains `motion.npz`, `terrain.obj`, `pair.json`, `meta.json`.

**Step 4: Short training smoke run**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
bash scripts/train.sh \
  --pair-manifest /tmp/tt_converted/platform_001/pair.json \
  --num-envs 8 \
  --max-iterations 1
```

Expected: env initializes, rollout collection starts, no motion-format or terrain-format error occurs immediately.

**Step 5: Commit**

If code changes were required during manual validation:

```bash
git add <fixed files>
git commit -m "fix: stabilize real-sample pair conversion"
```

If no code changes were required, skip commit for this task.
