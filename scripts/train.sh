#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "${SCRIPT_DIR}/lib/common.sh"

TASK="${TASK:-TT-Tracking-TerrainBlind-Unitree-G1}"
PAIR_MANIFEST="${PAIR_MANIFEST:-/tmp/tt_converted/beyond_dash_vault_001_aug001_dm/pair.json}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-tt_single_pair_beyond_dash_vault_001_aug001_dm}"
RUN_NAME="${RUN_NAME:-beyond_dash_vault_001_aug001_dm_g1_blind_primitive_boxes_n32768_it10000}"
COLLISION_BACKEND="${COLLISION_BACKEND:-primitive_boxes}"
SAMPLING_MODE="${SAMPLING_MODE:-start}"
NUM_ENVS="${NUM_ENVS:-32768}"
ENV_SPACING="${ENV_SPACING:-12.0}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU_IDS="${GPU_IDS:-[0]}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/data/junsong/.secrets/wandb.env}"

tt_train_exp "$@"
