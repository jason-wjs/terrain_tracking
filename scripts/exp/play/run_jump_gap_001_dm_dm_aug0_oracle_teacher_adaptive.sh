#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

## oracle teacher run_jump_gap_001_dm_dm_aug0 playback for adaptive sampling mode
TASK="${TASK:-TT-Tracking-TerrainOracleTeacher-Unitree-G1}" \
PAIR_MANIFEST="${PAIR_MANIFEST:-/tmp/tt_converted/run_jump_gap_001_dm_dm_aug0/pair.json}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME:-tt_single_pair_run_jump_gap_001_dm_dm_aug0}" \
RUN_NAME="${RUN_NAME:-run_jump_gap_001_dm_dm_aug0_g1_oracle_teacher_n8192_adaptive}" \
COLLISION_BACKEND="${COLLISION_BACKEND:-primitive_boxes}" \
SAMPLING_MODE="${SAMPLING_MODE:-adaptive}" \
NUM_ENVS="${NUM_ENVS:-8192}" \
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}" \
"${SCRIPT_DIR}/../../play.sh" "$@"