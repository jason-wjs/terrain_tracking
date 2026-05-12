#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

## baseline vaulting experiment with start sampling mode
TASK="${TASK:-TT-Tracking-TerrainBlind-Unitree-G1}" \
PAIR_MANIFEST="${PAIR_MANIFEST:-/tmp/tt_converted/beyond_dash_vault_001_aug001_dm/pair.json}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME:-tt_single_pair_beyond_dash_vault_001_aug001_dm}" \
RUN_NAME="${RUN_NAME:-beyond_dash_vault_001_aug001_dm_g1_blind_primitive_boxes_n32768_start}" \
COLLISION_BACKEND="${COLLISION_BACKEND:-primitive_boxes}" \
SAMPLING_MODE="${SAMPLING_MODE:-start}" \
NUM_ENVS="${NUM_ENVS:-32768}" \
MAX_ITERATIONS="${MAX_ITERATIONS:-100000}" \
"${SCRIPT_DIR}/../train.sh" "$@"
