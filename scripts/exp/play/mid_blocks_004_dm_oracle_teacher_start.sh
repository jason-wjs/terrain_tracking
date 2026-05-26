#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "${SCRIPT_DIR}/../../lib/common.sh"

## oracle teacher mid_blocks_004_dm playback for start sampling mode
TASK="${TASK:-TT-Tracking-TerrainOracleTeacher-Unitree-G1}"
PAIR_MANIFEST="${PAIR_MANIFEST:-/tmp/tt_converted/mid_blocks_004_dm/pair.json}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-tt_single_pair_mid_blocks_004_dm}"
RUN_NAME="${RUN_NAME:-mid_blocks_004_dm_g1_oracle_teacher_n8192_start}"
COLLISION_BACKEND="${COLLISION_BACKEND:-primitive_boxes}"
SAMPLING_MODE="${SAMPLING_MODE:-start}"
NUM_ENVS="${NUM_ENVS:-16384}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
tt_play_exp "$@"
