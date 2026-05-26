#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "${SCRIPT_DIR}/../../lib/common.sh"

## oracle teacher climb_00_z_scale_1.0 playback for adaptive sampling mode
TASK="${TASK:-TT-Tracking-TerrainOracleTeacher-Unitree-G1}"
PAIR_MANIFEST="${PAIR_MANIFEST:-/tmp/tt_converted_omniretarget/climb_00_z_scale_1.0/pair.json}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-tt_single_pair_climb_00_z_scale_1_0}"
RUN_NAME="${RUN_NAME:-climb_00_z_scale_1_0_g1_oracle_teacher_n8192_adaptive}"
COLLISION_BACKEND="${COLLISION_BACKEND:-omniretarget_boxes}"
SAMPLING_MODE="${SAMPLING_MODE:-adaptive}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
tt_play_exp "$@"
