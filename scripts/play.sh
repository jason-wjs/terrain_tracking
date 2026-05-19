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

tt_cd_repo_root
tt_require_file "${PAIR_MANIFEST}" "Pair manifest"

if [[ "${AGENT}" != "zero" && "${AGENT}" != "random" ]]; then
  if [[ -z "${CHECKPOINT_FILE}" ]]; then
    if [[ -z "${CHECKPOINT_DIR}" ]]; then
      shopt -s nullglob
      matching_run_dirs=("${TT_REPO_ROOT}/logs/rsl_rl/${EXPERIMENT_NAME}"/*_"${RUN_NAME}")
      shopt -u nullglob

      if (( ${#matching_run_dirs[@]} == 0 )); then
        echo "No run directory found for run '${RUN_NAME}' under logs/rsl_rl/${EXPERIMENT_NAME}." >&2
        echo "Train first via scripts/exp/train/<experiment>.sh or set CHECKPOINT_DIR/CHECKPOINT_FILE explicitly." >&2
        exit 1
      fi

      IFS=$'\n' sorted_run_dirs=($(printf '%s\n' "${matching_run_dirs[@]}" | sort))
      unset IFS
      last_run_idx=$((${#sorted_run_dirs[@]} - 1))
      CHECKPOINT_DIR="${sorted_run_dirs[${last_run_idx}]}"
    fi

    if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
      echo "Checkpoint directory not found: ${CHECKPOINT_DIR}" >&2
      exit 1
    fi

    shopt -s nullglob
    matching_models=("${CHECKPOINT_DIR}"/model_*.pt)
    shopt -u nullglob

    if (( ${#matching_models[@]} == 0 )); then
      echo "No model_*.pt checkpoint found under '${CHECKPOINT_DIR}'." >&2
      echo "Set CHECKPOINT_FILE explicitly or wait for checkpoints to be saved." >&2
      exit 1
    fi

    IFS=$'\n' sorted_models=($(printf '%s\n' "${matching_models[@]}" | sort -V))
    unset IFS
    last_model_idx=$((${#sorted_models[@]} - 1))
    CHECKPOINT_FILE="${sorted_models[${last_model_idx}]}"
  fi

  tt_require_file "${CHECKPOINT_FILE}" "Checkpoint file"
  echo "Using checkpoint: ${CHECKPOINT_FILE}"
fi

play_args=(
  --task "${TASK}"
  --agent "${AGENT}"
  --viewer "${VIEWER}"
  --pair-manifest "${PAIR_MANIFEST}"
  --collision-backend "${COLLISION_BACKEND}"
  --no-terminations "${NO_TERMINATIONS}"
)

if [[ -n "${CHECKPOINT_FILE}" ]]; then
  play_args+=(--checkpoint-file "${CHECKPOINT_FILE}")
fi

if [[ -n "${PLAY_NUM_ENVS}" ]]; then
  play_args+=(--num-envs "${PLAY_NUM_ENVS}")
fi

if [[ -n "${ENV_SPACING}" ]]; then
  play_args+=(--env-spacing "${ENV_SPACING}")
fi

if [[ -n "${DEVICE}" ]]; then
  play_args+=(--device "${DEVICE}")
fi

uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.play \
  "${play_args[@]}" \
  "$@"
