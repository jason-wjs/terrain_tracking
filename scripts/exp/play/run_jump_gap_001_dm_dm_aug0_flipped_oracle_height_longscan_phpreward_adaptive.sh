#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "${SCRIPT_DIR}/../../lib/common.sh"

## quick playback for the OracleHeight long-scan experiment on run_jump_gap_001_dm_dm_aug0_flipped
TASK="${TASK:-TT-Tracking-TerrainOracleHeightLongScanPhpReward-Unitree-G1}"
PAIR_MANIFEST="${PAIR_MANIFEST:-/tmp/tt_converted/run_jump_gap_001_dm_dm_aug0_flipped/pair.json}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-tt_single_pair_run_jump_gap_001_dm_dm_aug0_flipped}"
RUN_NAME="${RUN_NAME:-run_jump_gap_001_dm_dm_aug0_flipped_g1_oracle_height_longscan_phpreward_n16384_adaptive}"
COLLISION_BACKEND="${COLLISION_BACKEND:-primitive_boxes}"
PLAY_NUM_ENVS="${PLAY_NUM_ENVS:-1}"
NO_TERMINATIONS="${NO_TERMINATIONS:-True}"
tt_play_exp "$@"
