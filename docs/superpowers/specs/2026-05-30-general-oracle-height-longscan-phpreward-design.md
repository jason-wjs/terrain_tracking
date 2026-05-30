# General Oracle Height Long-Scan PHP Reward Design

This design adds a general multi-pair task that extends
`TT-Tracking-TerrainOracleHeightLongScanPhpReward-Unitree-G1` in the same way
the current general teacher task extends the single-pair oracle teacher task.

## Goals

- Train one terrain-aware controller over a `Pair dataset` while preserving the
  single-pair `OracleHeightLongScanPhpReward` observation and reward semantics.
- Reuse the existing general terrain tracking runtime: `MultiMotionCommand`,
  `PairFrameSampler`, `PairTerrainBank`, and `apply_pair_dataset_to_env_cfg`.
- Add a mid_blocks training script parallel to
  `scripts/exp/train/mid_blocks_general_oracle_teacher_adaptive.sh`.
- Keep the existing single-pair tasks and current general oracle teacher task
  unchanged.

## Non-Goals

- Do not add pelvis teacher privileged observations.
- Do not change the current `OracleTeacher` general task.
- Do not introduce a new terrain bank, pair sampler, or dataset format.
- Do not add OmniRetarget-specific training scripts in this change.

## Base Task Semantics

The new task starts from
`unitree_g1_oracle_height_long_scan_php_reward_terrain_tracking_env_cfg()`.
Therefore it inherits:

- the blind terrain tracking base configuration;
- the long forward height scan sensor with `GridPatternCfg(size=(2.0, 0.7),
  resolution=0.1)`;
- `height_scan` observations for both actor and critic;
- PHP reward weights:
  - `motion_global_root_pos.weight = 1.0`
  - `motion_global_root_ori.weight = 1.0`
  - `self_collisions.weight = -0.5`

It must not inherit or add the oracle teacher pelvis terms:

- `reference_pelvis_pos_error_b`
- `reference_pelvis_ori_error_b`
- `pelvis_lin_vel`
- `pelvis_ang_vel`
- `pelvis_global_pos_w`
- `pelvis_global_lin_vel_w`

## Generalization Architecture

The implementation should extract the common "generalize a single-pair env cfg"
steps from the existing general teacher config:

1. Build a single-pair base env cfg.
2. Replace `cfg.commands["motion"]`, a `MotionCommandCfg`, with a
   `MultiMotionCommandCfg` while preserving the relevant command settings.
3. Add `out_of_tile_bounds` as a non-timeout termination.
4. If `pair_dataset` is supplied, call `apply_pair_dataset_to_env_cfg(cfg)` so
   the scene receives the packed `PairTerrainBank`.

The current general oracle teacher task should continue using this shared helper
but keep its teacher-specific observation replacement:

```text
pelvis_global_pos_w -> pelvis_pair_local_pos_w
```

The new height-longscan PHP reward general task should not do that replacement,
because its base task has no `pelvis_global_pos_w` observation.

## Task Registration

Add the task IDs:

```text
TT-Tracking-TerrainOracleHeightLongScanPhpRewardGeneral-Unitree-G1
TT-Tracking-TerrainOracleHeightLongScanPhpRewardGeneral-Unitree-G1-Play
```

The play variant should mirror the current general teacher play registration:
both `env_cfg` and `play_env_cfg` use the play config.

## RL Config

Add:

```text
unitree_g1_general_oracle_height_long_scan_php_reward_terrain_tracking_ppo_runner_cfg
```

It should reuse `unitree_g1_oracle_terrain_tracking_ppo_runner_cfg()` and only
override:

```text
experiment_name = "g1_general_oracle_height_longscan_phpreward_terrain_tracking"
```

Script-level `--agent.experiment-name` remains the source of the actual rsl_rl
log directory name for experiment runs.

## Training Script

Add:

```text
scripts/exp/train/mid_blocks_general_oracle_height_longscan_phpreward_adaptive.sh
```

Defaults:

```bash
TASK="${TASK:-TT-Tracking-TerrainOracleHeightLongScanPhpRewardGeneral-Unitree-G1}"
PAIR_DATASET="${PAIR_DATASET:-/home/humanoid/Downloads/Data/parc_initial_aug_g1/pair_dataset_mid_blocks.jsonl}"
DATASET_VALIDATE="${DATASET_VALIDATE:-fast}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-tt_general_pair_dataset_mid_blocks}"
RUN_NAME="${RUN_NAME:-mid_blocks_general_g1_oracle_height_longscan_phpreward_n16384_adaptive}"
SAMPLING_MODE="${SAMPLING_MODE:-adaptive}"
PAIR_SAMPLER_MODE="${PAIR_SAMPLER_MODE:-independent}"
NUM_ENVS="${NUM_ENVS:-16384}"
MAX_ITERATIONS="${MAX_ITERATIONS:-50000}"
```

The script should call `tt_train_general_pair_dataset_exp "$@"`.

## Tests

Extend config tests to verify:

- the new general height-longscan PHP reward cfg replaces the motion command
  with `MultiMotionCommandCfg`;
- actor and critic include `height_scan`;
- the height scan sensor keeps the long forward pattern `(2.0, 0.7)`;
- PHP reward weights remain `1.0`, `1.0`, and `-0.5`;
- oracle teacher pelvis terms are absent;
- `out_of_tile_bounds` is registered.

Extend registry tests to verify both new task IDs are registered.

Extend CLI/script tests to verify the new mid_blocks script points at the new
task and the mid_blocks pair dataset.

## Success Criteria

- Existing single-pair oracle tasks still register and keep their current
  semantics.
- Existing general oracle teacher tests still pass.
- The new task can be selected by the general training frontend.
- The new script is a direct mid_blocks dataset entry point for the new task.
