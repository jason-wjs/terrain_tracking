#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "${SCRIPT_DIR}/lib/common.sh"

TASK="${TASK:-TT-Tracking-TerrainOracleTeacher-Unitree-G1}"
AGENT="${AGENT:-trained}"
VIEWER="${VIEWER:-viser}"
PAIR_MANIFEST="${PAIR_MANIFEST:-/tmp/tt_converted/mid_blocks_004_dm/pair.json}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-tt_single_pair_mid_blocks_004_dm}"
RUN_NAME="${RUN_NAME:-mid_blocks_004_dm_g1_oracle_teacher_n8192_adaptive}"
COLLISION_BACKEND="${COLLISION_BACKEND:-primitive_boxes}"
NO_TERMINATIONS="${NO_TERMINATIONS:-True}"
PLAY_NUM_ENVS="${PLAY_NUM_ENVS:-}"
ENV_SPACING="${ENV_SPACING:-}"
DEVICE="${DEVICE:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-}"

tt_play_exp "$@"
