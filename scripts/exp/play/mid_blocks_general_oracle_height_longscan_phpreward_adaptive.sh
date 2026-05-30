#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "${SCRIPT_DIR}/../../lib/common.sh"

## general oracle height long-scan PHP reward playback over PARC mid_blocks pairs.
TASK="${TASK:-TT-Tracking-TerrainOracleHeightLongScanPhpRewardGeneral-Unitree-G1}"
AGENT="${AGENT:-trained}"
VIEWER="${VIEWER:-viser}"
PAIR_DATASET="${PAIR_DATASET:-/home/humanoid/Downloads/Data/parc_initial_aug_g1/pair_dataset_mid_blocks.jsonl}"
DATASET_VALIDATE="${DATASET_VALIDATE:-fast}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-tt_general_pair_dataset_mid_blocks}"
RUN_NAME="${RUN_NAME:-mid_blocks_general_g1_oracle_height_longscan_phpreward_n16384_adaptive}"
PAIR_SAMPLER_MODE="${PAIR_SAMPLER_MODE:-independent}"
PLAY_NUM_ENVS="${PLAY_NUM_ENVS:-1}"
MAX_PAIRS="${MAX_PAIRS:-1}"
NO_TERMINATIONS="${NO_TERMINATIONS:-True}"
DEVICE="${DEVICE:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-}"

tt_cd_repo_root
tt_require_file "${PAIR_DATASET}" "Pair dataset"

if [[ "${AGENT}" != "zero" && "${AGENT}" != "random" ]]; then
  if [[ -z "${CHECKPOINT_FILE}" ]]; then
    CHECKPOINT_FILE=$(tt_latest_checkpoint_file "${EXPERIMENT_NAME}" "${RUN_NAME}" "${CHECKPOINT_DIR}")
  fi
  tt_require_file "${CHECKPOINT_FILE}" "Checkpoint file"
  echo "Using checkpoint: ${CHECKPOINT_FILE}"
fi

play_args=(
  --task "${TASK}"
  --agent "${AGENT}"
  --viewer "${VIEWER}"
  --pair-dataset "${PAIR_DATASET}"
  --dataset-validate "${DATASET_VALIDATE}"
  --pair-sampler-mode "${PAIR_SAMPLER_MODE}"
  --num-envs "${PLAY_NUM_ENVS}"
  --max-pairs "${MAX_PAIRS}"
  --no-terminations "${NO_TERMINATIONS}"
)

if [[ -n "${CHECKPOINT_FILE}" ]]; then
  play_args+=(--checkpoint-file "${CHECKPOINT_FILE}")
fi

if [[ -n "${DEVICE}" ]]; then
  play_args+=(--device "${DEVICE}")
fi

uv run python -m terrain_tracking.tasks.general_terrain_tracking.scripts.play \
  "${play_args[@]}" \
  "$@"
