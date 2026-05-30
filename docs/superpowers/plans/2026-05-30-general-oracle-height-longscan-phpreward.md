# General Oracle Height Long-Scan PHP Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a general multi-pair task that extends `OracleHeightLongScanPhpReward` without adding teacher pelvis oracle observations, plus a mid_blocks training script for it.

**Architecture:** Reuse the current `general_terrain_tracking` package and extract the shared "generalize a single-pair env cfg" logic from the existing general teacher config. The new task starts from `unitree_g1_oracle_height_long_scan_php_reward_terrain_tracking_env_cfg()`, swaps in `MultiMotionCommandCfg`, adds `out_of_tile_bounds`, and uses the existing pair dataset frontend.

**Tech Stack:** Python, mjlab task registry, RSL-RL config, bash experiment scripts, pytest.

---

## File Structure

- Modify `src/terrain_tracking/tasks/general_terrain_tracking/config/g1/env_cfgs.py`
  - Add a shared helper that turns a single-pair terrain tracking env cfg into a general pair-dataset env cfg.
  - Keep current teacher-specific `pelvis_global_pos_w -> pelvis_pair_local_pos_w` behavior only in the teacher cfg.
  - Add `unitree_g1_general_oracle_height_long_scan_php_reward_terrain_tracking_env_cfg`.
- Modify `src/terrain_tracking/tasks/general_terrain_tracking/config/g1/rl_cfg.py`
  - Add `unitree_g1_general_oracle_height_long_scan_php_reward_terrain_tracking_ppo_runner_cfg`.
- Modify `src/terrain_tracking/tasks/general_terrain_tracking/config/g1/__init__.py`
  - Register `TT-Tracking-TerrainOracleHeightLongScanPhpRewardGeneral-Unitree-G1`.
  - Register `TT-Tracking-TerrainOracleHeightLongScanPhpRewardGeneral-Unitree-G1-Play`.
- Create `scripts/exp/train/mid_blocks_general_oracle_height_longscan_phpreward_adaptive.sh`
  - Mid_blocks pair dataset entry point for the new task.
- Modify `tests/test_general_terrain_tracking_config.py`
  - Add config tests for long scan, PHP reward weights, absence of teacher terms, motion command replacement, and tile termination.
- Modify `tests/test_registry.py`
  - Assert the two new task IDs register.
- Modify `tests/test_cli_help.py`
  - Assert the new script points to the new general task and mid_blocks dataset.

### Task 1: Write Failing General Height-Longscan Config Tests

**Files:**
- Modify: `tests/test_general_terrain_tracking_config.py`

- [ ] **Step 1: Add imports and helpers for the new config test**

Add `GridPatternCfg`, `RayCastSensorCfg`, and the new env cfg import. Also add a local copy of the teacher term names so this test does not import another test module.

```python
from mjlab.sensor import GridPatternCfg, RayCastSensorCfg

from terrain_tracking.tasks.general_terrain_tracking.config.g1.env_cfgs import (
  unitree_g1_general_oracle_height_long_scan_php_reward_terrain_tracking_env_cfg,
  unitree_g1_general_oracle_teacher_terrain_tracking_env_cfg,
)
```

Add this constant near the imports:

```python
ORACLE_TEACHER_TERMS = {
  "reference_pelvis_pos_error_b",
  "reference_pelvis_ori_error_b",
  "pelvis_lin_vel",
  "pelvis_ang_vel",
  "pelvis_global_pos_w",
  "pelvis_global_lin_vel_w",
}
```

Add this helper after imports/constants:

```python
def _terrain_scan(cfg):
  sensor_by_name = {sensor.name: sensor for sensor in cfg.scene.sensors or ()}
  return sensor_by_name["terrain_scan"]
```

- [ ] **Step 2: Add the failing test**

Append this test to `tests/test_general_terrain_tracking_config.py`:

```python
def test_general_oracle_height_long_scan_php_reward_cfg_preserves_base_semantics():
  cfg = unitree_g1_general_oracle_height_long_scan_php_reward_terrain_tracking_env_cfg()

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MultiMotionCommandCfg)
  assert motion_cmd.pair_dataset == ""
  assert motion_cmd.body_names
  assert motion_cmd.anchor_body_name

  terrain_scan = _terrain_scan(cfg)
  assert isinstance(terrain_scan, RayCastSensorCfg)
  assert isinstance(terrain_scan.pattern, GridPatternCfg)
  assert terrain_scan.pattern.size == (2.0, 0.7)
  assert terrain_scan.pattern.resolution == 0.1

  assert cfg.rewards["motion_global_root_pos"].weight == 1.0
  assert cfg.rewards["motion_global_root_ori"].weight == 1.0
  assert cfg.rewards["self_collisions"].weight == -0.5

  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    assert "height_scan" in terms
    assert ORACLE_TEACHER_TERMS.isdisjoint(terms)
    assert "pelvis_pair_local_pos_w" not in terms

  termination = cfg.terminations["out_of_tile_bounds"]
  assert isinstance(termination, TerminationTermCfg)
  assert termination.func is out_of_tile_bounds
  assert termination.time_out is False
```

- [ ] **Step 3: Run the targeted config test and verify it fails**

Run:

```bash
uv run pytest -q tests/test_general_terrain_tracking_config.py
```

Expected: FAIL during import because `unitree_g1_general_oracle_height_long_scan_php_reward_terrain_tracking_env_cfg` does not exist yet.

- [ ] **Step 4: Commit the failing test**

```bash
git add tests/test_general_terrain_tracking_config.py
git commit -m "test: cover general oracle height longscan config"
```

If the worktree contains unrelated changes, stage only `tests/test_general_terrain_tracking_config.py`.

### Task 2: Implement the General Height-Longscan Env Config

**Files:**
- Modify: `src/terrain_tracking/tasks/general_terrain_tracking/config/g1/env_cfgs.py`
- Test: `tests/test_general_terrain_tracking_config.py`

- [ ] **Step 1: Update imports**

Change the oracle env cfg import block to include the long-scan PHP reward base:

```python
from terrain_tracking.tasks.oracle_terrain_tracking.config.g1.env_cfgs import (
  unitree_g1_oracle_height_long_scan_php_reward_terrain_tracking_env_cfg,
  unitree_g1_oracle_teacher_terrain_tracking_env_cfg,
)
```

- [ ] **Step 2: Extract the shared generalization helper**

Insert this helper after `_replace_world_teacher_position_with_pair_local`:

```python
def _generalize_single_pair_env_cfg(
  cfg: ManagerBasedRlEnvCfg,
  *,
  pair_dataset: str,
) -> ManagerBasedRlEnvCfg:
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)

  cfg.commands["motion"] = MultiMotionCommandCfg(
    pair_dataset=pair_dataset,
    entity_name=motion_cmd.entity_name,
    resampling_time_range=motion_cmd.resampling_time_range,
    debug_vis=motion_cmd.debug_vis,
    anchor_body_name=motion_cmd.anchor_body_name,
    body_names=motion_cmd.body_names,
    pose_range=motion_cmd.pose_range,
    velocity_range=motion_cmd.velocity_range,
    joint_position_range=motion_cmd.joint_position_range,
    sampling_mode=motion_cmd.sampling_mode,
  )
  cfg.terminations["out_of_tile_bounds"] = TerminationTermCfg(
    func=mdp.out_of_tile_bounds,
    params={"command_name": "motion", "fail_margin": 1.0},
    time_out=False,
  )
  if pair_dataset:
    apply_pair_dataset_to_env_cfg(cfg)
  return cfg
```

- [ ] **Step 3: Refactor the existing teacher cfg to use the helper**

Replace the body of `unitree_g1_general_oracle_teacher_terrain_tracking_env_cfg` with:

```python
def unitree_g1_general_oracle_teacher_terrain_tracking_env_cfg(
  *,
  pair_dataset: str = "",
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_oracle_teacher_terrain_tracking_env_cfg(play=play)
  _generalize_single_pair_env_cfg(cfg, pair_dataset=pair_dataset)
  _replace_world_teacher_position_with_pair_local(cfg)
  return cfg
```

- [ ] **Step 4: Add the new height-longscan PHP reward cfg**

Add this function below the teacher cfg:

```python
def unitree_g1_general_oracle_height_long_scan_php_reward_terrain_tracking_env_cfg(
  *,
  pair_dataset: str = "",
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_oracle_height_long_scan_php_reward_terrain_tracking_env_cfg(
    play=play
  )
  return _generalize_single_pair_env_cfg(cfg, pair_dataset=pair_dataset)
```

- [ ] **Step 5: Update `__all__`**

Replace the existing `__all__` with:

```python
__all__ = [
  "unitree_g1_general_oracle_height_long_scan_php_reward_terrain_tracking_env_cfg",
  "unitree_g1_general_oracle_teacher_terrain_tracking_env_cfg",
]
```

- [ ] **Step 6: Run the targeted config tests**

Run:

```bash
uv run pytest -q tests/test_general_terrain_tracking_config.py
```

Expected: PASS.

- [ ] **Step 7: Commit the env config implementation**

```bash
git add src/terrain_tracking/tasks/general_terrain_tracking/config/g1/env_cfgs.py
git commit -m "feat: add general oracle height longscan env config"
```

### Task 3: Add RL Config and Task Registration

**Files:**
- Modify: `src/terrain_tracking/tasks/general_terrain_tracking/config/g1/rl_cfg.py`
- Modify: `src/terrain_tracking/tasks/general_terrain_tracking/config/g1/__init__.py`
- Modify: `tests/test_registry.py`

- [ ] **Step 1: Add failing registry assertions**

In `tests/test_registry.py`, extend `test_general_oracle_teacher_task_registers_with_mjlab_registry`:

```python
  assert "TT-Tracking-TerrainOracleHeightLongScanPhpRewardGeneral-Unitree-G1" in tasks
  assert (
    "TT-Tracking-TerrainOracleHeightLongScanPhpRewardGeneral-Unitree-G1-Play"
    in tasks
  )
```

- [ ] **Step 2: Run the registry test and verify it fails**

Run:

```bash
uv run pytest -q tests/test_registry.py::test_general_oracle_teacher_task_registers_with_mjlab_registry
```

Expected: FAIL because the new task IDs are not registered.

- [ ] **Step 3: Add the new RL cfg function**

In `src/terrain_tracking/tasks/general_terrain_tracking/config/g1/rl_cfg.py`, add:

```python
def unitree_g1_general_oracle_height_long_scan_php_reward_terrain_tracking_ppo_runner_cfg() -> (
  RslRlOnPolicyRunnerCfg
):
  cfg = unitree_g1_oracle_terrain_tracking_ppo_runner_cfg()
  cfg.experiment_name = (
    "g1_general_oracle_height_longscan_phpreward_terrain_tracking"
  )
  return cfg
```

Replace `__all__` with:

```python
__all__ = [
  "unitree_g1_general_oracle_height_long_scan_php_reward_terrain_tracking_ppo_runner_cfg",
  "unitree_g1_general_oracle_teacher_terrain_tracking_ppo_runner_cfg",
]
```

- [ ] **Step 4: Register the new task IDs**

In `src/terrain_tracking/tasks/general_terrain_tracking/config/g1/__init__.py`, update imports:

```python
from .env_cfgs import (
  unitree_g1_general_oracle_height_long_scan_php_reward_terrain_tracking_env_cfg,
  unitree_g1_general_oracle_teacher_terrain_tracking_env_cfg,
)
from .rl_cfg import (
  unitree_g1_general_oracle_height_long_scan_php_reward_terrain_tracking_ppo_runner_cfg,
  unitree_g1_general_oracle_teacher_terrain_tracking_ppo_runner_cfg,
)
```

Add these registrations after the existing teacher registrations:

```python
register_mjlab_task(
  task_id="TT-Tracking-TerrainOracleHeightLongScanPhpRewardGeneral-Unitree-G1",
  env_cfg=unitree_g1_general_oracle_height_long_scan_php_reward_terrain_tracking_env_cfg(),
  play_env_cfg=unitree_g1_general_oracle_height_long_scan_php_reward_terrain_tracking_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_general_oracle_height_long_scan_php_reward_terrain_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="TT-Tracking-TerrainOracleHeightLongScanPhpRewardGeneral-Unitree-G1-Play",
  env_cfg=unitree_g1_general_oracle_height_long_scan_php_reward_terrain_tracking_env_cfg(
    play=True
  ),
  play_env_cfg=unitree_g1_general_oracle_height_long_scan_php_reward_terrain_tracking_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_general_oracle_height_long_scan_php_reward_terrain_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
```

- [ ] **Step 5: Run registry and config tests**

Run:

```bash
uv run pytest -q tests/test_registry.py::test_general_oracle_teacher_task_registers_with_mjlab_registry tests/test_general_terrain_tracking_config.py
```

Expected: PASS.

- [ ] **Step 6: Commit registration and RL config**

```bash
git add src/terrain_tracking/tasks/general_terrain_tracking/config/g1/rl_cfg.py src/terrain_tracking/tasks/general_terrain_tracking/config/g1/__init__.py tests/test_registry.py
git commit -m "feat: register general oracle height longscan task"
```

### Task 4: Add the Mid-Blocks Training Script

**Files:**
- Create: `scripts/exp/train/mid_blocks_general_oracle_height_longscan_phpreward_adaptive.sh`
- Modify: `tests/test_cli_help.py`

- [ ] **Step 1: Add failing CLI/script assertions**

In `tests/test_cli_help.py`, update `test_general_mid_blocks_exp_scripts_target_pair_dataset_training` by reading the new script:

```python
  height_longscan_train_script = (
    scripts_root
    / "exp"
    / "train"
    / "mid_blocks_general_oracle_height_longscan_phpreward_adaptive.sh"
  ).read_text(encoding="utf-8")
```

Add these assertions after the existing general teacher script assertions:

```python
  assert (
    "TT-Tracking-TerrainOracleHeightLongScanPhpRewardGeneral-Unitree-G1"
    in height_longscan_train_script
  )
  assert "pair_dataset_mid_blocks.jsonl" in height_longscan_train_script
  assert "mid_blocks_general_g1_oracle_height_longscan_phpreward_n16384_adaptive" in height_longscan_train_script
  assert "PAIR_SAMPLER_MODE=\"${PAIR_SAMPLER_MODE:-independent}\"" in height_longscan_train_script
  assert "tt_train_general_pair_dataset_exp" in height_longscan_train_script
```

- [ ] **Step 2: Run the CLI test and verify it fails**

Run:

```bash
uv run pytest -q tests/test_cli_help.py::test_general_mid_blocks_exp_scripts_target_pair_dataset_training
```

Expected: FAIL with `FileNotFoundError` for the new script.

- [ ] **Step 3: Create the script**

Create `scripts/exp/train/mid_blocks_general_oracle_height_longscan_phpreward_adaptive.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "${SCRIPT_DIR}/../../lib/common.sh"

## general oracle height long-scan PHP reward over all PARC mid_blocks pairs.
TASK="${TASK:-TT-Tracking-TerrainOracleHeightLongScanPhpRewardGeneral-Unitree-G1}"
PAIR_DATASET="${PAIR_DATASET:-/home/humanoid/Downloads/Data/parc_initial_aug_g1/pair_dataset_mid_blocks.jsonl}"
DATASET_VALIDATE="${DATASET_VALIDATE:-fast}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-tt_general_pair_dataset_mid_blocks}"
RUN_NAME="${RUN_NAME:-mid_blocks_general_g1_oracle_height_longscan_phpreward_n16384_adaptive}"
SAMPLING_MODE="${SAMPLING_MODE:-adaptive}"
PAIR_SAMPLER_MODE="${PAIR_SAMPLER_MODE:-independent}"
NUM_ENVS="${NUM_ENVS:-16384}"
MAX_ITERATIONS="${MAX_ITERATIONS:-50000}"

tt_train_general_pair_dataset_exp "$@"
```

- [ ] **Step 4: Mark the script executable**

Run:

```bash
chmod +x scripts/exp/train/mid_blocks_general_oracle_height_longscan_phpreward_adaptive.sh
```

- [ ] **Step 5: Run the CLI/script tests**

Run:

```bash
uv run pytest -q tests/test_cli_help.py::test_exp_scripts_source_common_library tests/test_cli_help.py::test_general_mid_blocks_exp_scripts_target_pair_dataset_training
```

Expected: PASS.

- [ ] **Step 6: Commit the script and test**

```bash
git add scripts/exp/train/mid_blocks_general_oracle_height_longscan_phpreward_adaptive.sh tests/test_cli_help.py
git commit -m "feat: add mid blocks general height longscan script"
```

### Task 5: Final Verification

**Files:**
- No code changes unless verification finds a real issue.

- [ ] **Step 1: Run focused tests**

Run:

```bash
uv run pytest -q tests/test_general_terrain_tracking_config.py tests/test_registry.py tests/test_cli_help.py
```

Expected: PASS.

- [ ] **Step 2: Run a broader relevant subset**

Run:

```bash
uv run pytest -q tests/test_oracle_terrain_tracking_config.py tests/test_general_env_smoke.py tests/test_pair_dataset.py tests/test_pair_terrain_bank.py tests/test_pair_frame_sampler.py
```

Expected: PASS. If this fails because local MuJoCo/GPU runtime is unavailable, record the exact failure and run the narrower non-runtime tests that still exercise the config and registry changes.

- [ ] **Step 3: Check git status**

Run:

```bash
git status --short
```

Expected: only unrelated pre-existing files may remain modified. The files touched by this plan should either be committed or intentionally staged for the current implementation work.

- [ ] **Step 4: Summarize the result**

Report:

```text
Implemented TT-Tracking-TerrainOracleHeightLongScanPhpRewardGeneral-Unitree-G1 and -Play.
Added scripts/exp/train/mid_blocks_general_oracle_height_longscan_phpreward_adaptive.sh.
Verified with: <commands run>.
Remaining unrelated worktree changes: <summary from git status>.
```

## Self-Review

- Spec coverage: The plan covers the new base semantics, shared helper extraction, task IDs, RL config, mid_blocks script, and tests.
- Placeholder scan: No placeholder markers or vague fill-in instructions are present.
- Type consistency: Function names, task IDs, script names, and runner cfg names match the approved spec.
