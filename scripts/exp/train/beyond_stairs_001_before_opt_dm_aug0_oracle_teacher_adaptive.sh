#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "${SCRIPT_DIR}/../../lib/common.sh"

## before_opt dm aug0 experiment with adaptive sampling mode
TASK="${TASK:-TT-Tracking-TerrainOracleTeacher-Unitree-G1}"
PAIR_MANIFEST="${PAIR_MANIFEST:-/tmp/tt_converted/beyond_stairs_001_before_opt_dm_aug0/pair.json}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-tt_single_pair_beyond_stairs_001_before_opt_dm_aug0}"
RUN_NAME="${RUN_NAME:-beyond_stairs_001_before_opt_dm_aug0_g1_oracle_teacher_adaptive_n8192}"
COLLISION_BACKEND="${COLLISION_BACKEND:-primitive_boxes}"
SAMPLING_MODE="${SAMPLING_MODE:-adaptive}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30000}"
tt_train_exp "$@"