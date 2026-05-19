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
MAX_ITERATIONS="${MAX_ITERATIONS:-100000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
GPU_IDS="${GPU_IDS:-[0]}"
WANDB_ENV_FILE="${WANDB_ENV_FILE:-/data/junsong/.secrets/wandb.env}"

tt_cd_repo_root
tt_source_wandb_env "${WANDB_ENV_FILE}"
tt_require_file "${PAIR_MANIFEST}" "Pair manifest"
tt_export_cuda "${CUDA_VISIBLE_DEVICES}" "${GPU_IDS}"

uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.train \
  --task "${TASK}" \
  --agent.experiment-name "${EXPERIMENT_NAME}" \
  --agent.run-name "${RUN_NAME}" \
  --collision-backend "${COLLISION_BACKEND}" \
  --env.commands.motion.sampling-mode "${SAMPLING_MODE}" \
  --pair-manifest "${PAIR_MANIFEST}" \
  --env.scene.num-envs "${NUM_ENVS}" \
  --env.scene.env-spacing 12.0 \
  --agent.max-iterations "${MAX_ITERATIONS}" \
  --gpu-ids "${GPU_IDS}" \
  "$@"
