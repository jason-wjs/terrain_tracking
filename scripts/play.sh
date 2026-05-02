#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)
cd "${REPO_ROOT}"

TASK="${TASK:-TT-Tracking-TerrainBlind-Unitree-G1}"
AGENT="${AGENT:-trained}"
VIEWER="${VIEWER:-native}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-tt_single_pair_platform_001}"
RUN_NAME="${RUN_NAME:-platform_001_g1_blind_NoPlaneCollision_start_n8192_it10000}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-}"

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  shopt -s nullglob
  matching_run_dirs=("${REPO_ROOT}/logs/rsl_rl/${EXPERIMENT_NAME}"/*_"${RUN_NAME}")
  shopt -u nullglob

  if (( ${#matching_run_dirs[@]} == 0 )); then
    echo "No run directory found for run '${RUN_NAME}' under logs/rsl_rl/${EXPERIMENT_NAME}." >&2
    echo "Train first via scripts/train.sh or set CHECKPOINT_DIR explicitly." >&2
    exit 1
  fi

  IFS=$'\n' sorted_run_dirs=($(printf '%s\n' "${matching_run_dirs[@]}" | sort))
  unset IFS
  last_run_idx=$((${#sorted_run_dirs[@]} - 1))
  CHECKPOINT_DIR="${sorted_run_dirs[${last_run_idx}]}"
fi

if [[ -z "${CHECKPOINT_FILE}" ]]; then
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

## visualizing
# uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.play \
#   --task "${TASK}" \
#   --agent zero \
#   --viewer "${VIEWER}" \
#   --pair-manifest /tmp/tt_converted/platform_001/pair.json \
#   --no-terminations True \
#   --num-envs 4 \
#   "$@"


# # platform_001
# pair manifest: /tmp/tt_converted/platform_001/pair.json
uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.play \
  --task "${TASK}" \
  --agent "${AGENT}" \
  --viewer "${VIEWER}" \
  --pair-manifest /tmp/tt_converted/platform_001/pair.json \
  --checkpoint-file /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/logs/rsl_rl/tt_single_pair_platform_001/2026-04-28_21-57-13_platform_001_g1_blind_NoPlaneCollision_start_n8192_it10000/model_9000.pt \
  --no-terminations True \
  "$@"

## mid_blocks_004_dm
## pair manifest: /tmp/tt_converted/mid_blocks_004_dm/pair.json
# uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.play \
#   --task "${TASK}" \
#   --agent "${AGENT}" \
#   --viewer "${VIEWER}" \
#   --pair-manifest /tmp/tt_converted/mid_blocks_004_dm/pair.json \
#   --checkpoint-file /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/logs/rsl_rl/tt_single_pair_platform_001/2026-04-29_23-10-24_platform_001_g1_blind_NoPlaneCollision_AS_n8192_it10000/model_9500.pt \
#   --no-terminations True \
#   "$@"

