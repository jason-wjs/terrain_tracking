#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "${SCRIPT_DIR}/../../lib/common.sh"

## OracleHeight long-scan experiment on mdm_castle_stairs_dm
TASK="${TASK:-TT-Tracking-TerrainOracleHeightLongScanPhpReward-Unitree-G1}"
PAIR_MANIFEST="${PAIR_MANIFEST:-/tmp/tt_converted/mdm_castle_stairs_dm/pair.json}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-tt_single_pair_mdm_castle_stairs_dm}"
RUN_NAME="${RUN_NAME:-mdm_castle_stairs_dm_g1_oracle_height_longscan_phpreward_n16384_adaptive}"
COLLISION_BACKEND="${COLLISION_BACKEND:-primitive_boxes}"
SAMPLING_MODE="${SAMPLING_MODE:-adaptive}"
NUM_ENVS="${NUM_ENVS:-16384}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
tt_train_exp "$@"
