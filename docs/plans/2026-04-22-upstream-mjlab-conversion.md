# Upstream MJLab Conversion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move motion conversion into `holosoma_retargeting`, keep `terrain_tracking` as a thin pair-manifest consumer, and validate the new flow on `platform_001` and `mid_blocks_004_dm`.

**Architecture:** `holosoma_retargeting` will own the conversion from retargeted `qpos.npz` into `mjlab`-compatible `motion.npz`, using an explicitly supplied MuJoCo robot XML asset. `terrain_tracking` will stop owning motion conversion and will only compose `pair.json` that references a converted motion file, a terrain mesh, and three terrain correction parameters.

**Tech Stack:** Python 3.13, NumPy, MuJoCo Python bindings, `tyro`, `pytest`, existing `holosoma_retargeting` config/CLI pattern, `terrain_tracking` runtime pair manifest.

---

### Task 1: Freeze The Downstream Contract

**Files:**
- Modify: `terrain_tracking/tests/test_pair_manifest.py`
- Modify: `terrain_tracking/tests/test_data_conversion_convert_pair.py`
- Modify: `terrain_tracking/tests/test_cli_help.py`
- Reference: `terrain_tracking/src/terrain_tracking/runtime/pair_manifest.py`
- Reference: `terrain_tracking/src/terrain_tracking/runtime/apply_pair.py`

**Step 1: Write the failing test**

Add or update tests so the downstream contract is explicit:

- `pair.json` requires `motion_file` and `terrain_file`
- `terrain_translation`, `terrain_quat_xyzw`, `terrain_scale` remain optional with defaults
- the compose script must not inspect retargeted `qpos`
- the compose script must preserve user-provided relative paths and terrain correction values

Example assertion shape:

```python
manifest = json.loads(output_path.read_text())
assert manifest["motion_file"] == "motion.npz"
assert manifest["terrain_file"] == "terrain.obj"
assert manifest["terrain_translation"] == [0.0, 0.0, 0.0]
```

**Step 2: Run test to verify it fails**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
pytest tests/test_pair_manifest.py tests/test_data_conversion_convert_pair.py tests/test_cli_help.py -v
```

Expected: FAIL because current `convert_pair` still assumes retarget input and motion conversion.

**Step 3: Write minimal implementation target**

Do not change runtime files yet. Only document the contract in tests so later refactor cannot accidentally re-expand downstream responsibilities.

**Step 4: Run test to verify the failure is stable**

Run the same command again and confirm failure is only in the intended assertions.

**Step 5: Commit**

```bash
git -C /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking add tests/test_pair_manifest.py tests/test_data_conversion_convert_pair.py tests/test_cli_help.py
git -C /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking commit -m "test: freeze pair manifest contract"
```


### Task 2: Extend Upstream Conversion Config And CLI

**Files:**
- Modify: `RETARGET/holosoma/src/holosoma_retargeting/holosoma_retargeting/config_types/data_conversion.py`
- Modify: `RETARGET/holosoma/src/holosoma_retargeting/holosoma_retargeting/data_conversion/convert_data_format_mj.py`
- Create: `RETARGET/holosoma/src/holosoma_retargeting/holosoma_retargeting/tests/test_data_conversion_mjlab_cli.py`
- Optional Create: `RETARGET/holosoma/scripts/retargeting/convert_parc_to_mjlab.sh`

**Step 1: Write the failing test**

Add upstream tests for CLI/config parsing that assert support for this workflow:

- input is a retargeted `qpos.npz`
- output is an `mjlab` tracking motion `.npz`
- robot asset can be overridden by explicit path
- output format is selectable as `mjlab_tracking`
- no terrain input is required for motion conversion

Example assertions:

```python
cfg = DataConversionConfig(
    input_file="clip.npz",
    output_name="clip_mjlab.npz",
    robot="g1",
)
assert cfg.output_fps == 50
```

**Step 2: Run test to verify it fails**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma/src/holosoma_retargeting
pytest holosoma_retargeting/tests/test_data_conversion_mjlab_cli.py -v
```

Expected: FAIL because current config/CLI does not cleanly represent the new upstream-only contract.

**Step 3: Write minimal implementation**

Update `DataConversionConfig` and `convert_data_format_mj.py` so the public entrypoint supports:

- `--input-file`
- `--output-name` or output path
- `--robot`
- `--robot-config.robot-urdf-file` or equivalent explicit robot XML/URDF override
- `--output-fps`
- `--once`
- `--output-format mjlab_tracking`

Keep one public conversion script. Do not introduce multiple public conversion CLIs.

**Step 4: Run test to verify it passes**

Run the same pytest command and confirm PASS.

**Step 5: Commit**

```bash
git -C /home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma add \
  src/holosoma_retargeting/holosoma_retargeting/config_types/data_conversion.py \
  src/holosoma_retargeting/holosoma_retargeting/data_conversion/convert_data_format_mj.py \
  src/holosoma_retargeting/holosoma_retargeting/tests/test_data_conversion_mjlab_cli.py
git -C /home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma commit -m "feat: add upstream mjlab conversion entrypoint"
```


### Task 3: Make Upstream Conversion Emit Correct MJLab Motion NPZ

**Files:**
- Modify: `RETARGET/holosoma/src/holosoma_retargeting/holosoma_retargeting/data_conversion/convert_data_format_mj.py`
- Create: `RETARGET/holosoma/src/holosoma_retargeting/holosoma_retargeting/tests/test_data_conversion_mjlab_output.py`
- Reference: `terrain_tracking/src/terrain_tracking/runtime/pair_manifest.py`
- Reference: `terrain_tracking/src/terrain_tracking/tasks/blind_terrain_tracking/scripts/common.py`

**Step 1: Write the failing test**

Add upstream output tests using a tiny synthetic retargeted `qpos.npz` fixture. Assert the generated `.npz` contains:

- `fps`
- `joint_pos`
- `joint_vel`
- `body_pos_w`
- `body_quat_w`
- `body_lin_vel_w`
- `body_ang_vel_w`

Also assert:

- frame count matches resampled output
- `joint_pos.shape[1]` matches robot DOF count
- quaternions are normalized
- body arrays share identical leading time dimension

**Step 2: Run test to verify it fails**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma/src/holosoma_retargeting
pytest holosoma_retargeting/tests/test_data_conversion_mjlab_output.py -v
```

Expected: FAIL because current script is holosoma-oriented and does not yet guarantee the downstream `mjlab` contract.

**Step 3: Write minimal implementation**

Inside `convert_data_format_mj.py`, adapt the existing logic so that for `mjlab_tracking` export it:

- loads retargeted `qpos`
- resamples to output FPS
- computes root/joint velocities
- loads the supplied robot asset with MuJoCo
- performs FK/body state extraction in asset order
- writes `mjlab`-compatible fields only

Do not add `pair.json` writing here. Upstream stops at `motion.npz`.

**Step 4: Run test to verify it passes**

Run the same pytest command and confirm PASS.

**Step 5: Commit**

```bash
git -C /home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma add \
  src/holosoma_retargeting/holosoma_retargeting/data_conversion/convert_data_format_mj.py \
  src/holosoma_retargeting/holosoma_retargeting/tests/test_data_conversion_mjlab_output.py
git -C /home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma commit -m "feat: emit mjlab tracking motion from retargeted qpos"
```


### Task 4: Collapse Downstream Conversion Into Pair Composition Only

**Files:**
- Modify: `terrain_tracking/src/terrain_tracking/data_conversion/convert_pair.py`
- Modify: `terrain_tracking/src/terrain_tracking/data_conversion/pair_bundle.py`
- Delete: `terrain_tracking/src/terrain_tracking/data_conversion/retarget_input.py`
- Delete: `terrain_tracking/src/terrain_tracking/data_conversion/mjlab_motion.py`
- Modify: `terrain_tracking/tests/test_data_conversion_convert_pair.py`
- Delete or Rewrite: `terrain_tracking/tests/test_data_conversion_retarget_input.py`
- Modify: `terrain_tracking/scripts/convert_pair.sh`

**Step 1: Write the failing test**

Add tests that treat the downstream compose step as pure manifest composition:

- input: existing converted `motion.npz`
- input: existing `terrain.obj`
- input: terrain translation/quaternion/scale
- output: `pair.json` and optional `meta.json`

The test should explicitly fail if the script tries to open retargeted `qpos` or compute motion arrays.

**Step 2: Run test to verify it fails**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
pytest tests/test_data_conversion_convert_pair.py tests/test_cli_help.py -v
```

Expected: FAIL because current implementation still owns motion conversion.

**Step 3: Write minimal implementation**

Refactor downstream so `convert_pair.py` only:

- validates that `motion.npz` exists
- validates that `terrain.obj` exists
- accepts terrain correction parameters
- writes `pair.json`
- optionally writes `meta.json`

Update `scripts/convert_pair.sh` to call the new thin compose flow and keep the hardcoded usage style.

**Step 4: Run test to verify it passes**

Run the same pytest command and confirm PASS.

**Step 5: Commit**

```bash
git -C /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking add \
  src/terrain_tracking/data_conversion/convert_pair.py \
  src/terrain_tracking/data_conversion/pair_bundle.py \
  scripts/convert_pair.sh \
  tests/test_data_conversion_convert_pair.py \
  tests/test_cli_help.py
git -C /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking rm \
  src/terrain_tracking/data_conversion/retarget_input.py \
  src/terrain_tracking/data_conversion/mjlab_motion.py \
  tests/test_data_conversion_retarget_input.py
git -C /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking commit -m "refactor: reduce conversion to pair manifest composition"
```


### Task 5: Add End-To-End Scripts And Documentation

**Files:**
- Modify: `RETARGET/holosoma/scripts/retargeting/run_parc_process.sh`
- Modify: `RETARGET/holosoma/scripts/retargeting/vis_parc_process.sh`
- Optional Create: `RETARGET/holosoma/scripts/retargeting/convert_parc_to_mjlab.sh`
- Modify: `terrain_tracking/scripts/convert_pair.sh`
- Modify: `terrain_tracking/docs/plans/2026-04-22-upstream-mjlab-conversion.md`
- Optional Modify: `RETARGET/holosoma/docs/plans/2026-04-20-parc-process-bootstrap.md`

**Step 1: Write the failing test or executable check**

For shell entrypoints, use executable smoke checks instead of unit tests:

- `bash -n` on each shell script
- `--help` smoke checks where available

**Step 2: Run check to verify current gaps**

Run:

```bash
bash -n /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/scripts/convert_pair.sh
bash -n /home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma/scripts/retargeting/run_parc_process.sh
bash -n /home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma/scripts/retargeting/vis_parc_process.sh
```

Expected: scripts may parse, but there is no single documented path from retarget result to converted `mjlab` motion plus downstream pair composition.

**Step 3: Write minimal implementation**

Document and script the full command sequence:

1. retarget / export terrain upstream
2. convert retargeted `qpos.npz` to `mjlab` `motion.npz` upstream
3. compose `pair.json` downstream
4. train/play downstream

Keep shell scripts intentionally hardcoded and user-editable.

**Step 4: Run syntax checks**

Run the same `bash -n` commands and confirm success.

**Step 5: Commit**

```bash
git -C /home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma add scripts/retargeting
git -C /home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma commit -m "docs: add upstream conversion shell entrypoints"
git -C /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking add scripts/convert_pair.sh docs/plans/2026-04-22-upstream-mjlab-conversion.md
git -C /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking commit -m "docs: document pair composition flow"
```


### Task 6: Validate Two Real Samples End To End

**Files:**
- Use: `/tmp/parc_process_workspace/retargeted/platform_001_original.npz`
- Use: `/tmp/parc_process_workspace/workspace/platform_001/platform_terrain.obj` or actual exported terrain path
- Use: `/tmp/parc_process_workspace/retargeted/mid_blocks_004_dm_original.npz`
- Use: `/tmp/parc_process_workspace/workspace/mid_blocks_004_dm/multi_boxes_scaled_0.74_0.74_0.74.obj` or actual exported terrain path
- Output: `/tmp/tt_converted/platform_001`
- Output: `/tmp/tt_converted/mid_blocks_004_dm`

**Step 1: Run upstream conversion on `platform_001`**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma
source scripts/source_retargeting_setup.sh
python src/holosoma_retargeting/holosoma_retargeting/data_conversion/convert_data_format_mj.py \
  --input-file /tmp/parc_process_workspace/retargeted/platform_001_original.npz \
  --output-name /tmp/tt_converted/platform_001/motion.npz \
  --robot g1 \
  --output-format mjlab_tracking \
  --robot-config.robot-urdf-file /absolute/path/to/mjlab/g1_asset.xml
```

Expected: `motion.npz` written with required `mjlab` fields.

**Step 2: Compose downstream pair for `platform_001`**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
python -m terrain_tracking.data_conversion.convert_pair \
  --motion-file /tmp/tt_converted/platform_001/motion.npz \
  --terrain-file /tmp/parc_process_workspace/workspace/platform_001/platform_terrain.obj \
  --output-dir /tmp/tt_converted/platform_001 \
  --sample-name platform_001 \
  --terrain-translation 0 0 0 \
  --terrain-quat-xyzw 0 0 0 1 \
  --terrain-scale 1 1 1
```

Expected: `pair.json` exists and references the correct files.

**Step 3: Repeat for `mid_blocks_004_dm`**

Run the equivalent two commands for `mid_blocks_004_dm`.

**Step 4: Run downstream smoke checks**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
pytest tests/test_pair_manifest.py tests/test_env_smoke.py -v
python scripts/play.sh --pair-manifest /tmp/tt_converted/platform_001/pair.json
python scripts/play.sh --pair-manifest /tmp/tt_converted/mid_blocks_004_dm/pair.json
```

Expected:

- unit tests PASS
- both pair manifests load
- play opens the expected terrain and motion pair without format errors

**Step 5: Manual acceptance**

Visually confirm:

- robot pose ordering is correct
- no obvious global frame flips
- terrain placement matches expected semantics
- results are no worse than current relaxed retarget baseline before training

**Step 6: Final commit**

Commit only after both samples convert and play successfully.


### Task 7: Remove Dead Downstream Assumptions

**Files:**
- Search and modify as needed under `terrain_tracking/src/terrain_tracking`
- Search and modify as needed under `terrain_tracking/tests`

**Step 1: Search for old assumptions**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
rg -n "retarget_input|emit_mjlab_motion|qpos|output_fps" src tests scripts
```

**Step 2: Remove stale references**

Delete or update any leftover references that imply `terrain_tracking` still converts retargeted motion itself.

**Step 3: Run full downstream test suite**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
pytest tests -v
```

Expected: PASS, except for tests that are already known to depend on unavailable simulator/runtime resources. Document any such skips explicitly.

**Step 4: Final verification sweep**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma/src/holosoma_retargeting
pytest holosoma_retargeting/tests -v
```

Expected: PASS for the new conversion tests and no regression in existing `parc_process` tests.

**Step 5: Final integration note**

Record the final command sequence and any required environment assumptions in the relevant shell scripts or docs before opening review.
