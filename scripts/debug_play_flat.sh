#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)
cd "${REPO_ROOT}"

TASK="Mjlab-Tracking-Flat-Unitree-G1"
AGENT="trained"
VIEWER="${VIEWER:-viser}"
MOTION_FILE="/tmp/tt_converted/platform_001/motion.npz"
CHECKPOINT_DIR="/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/logs/rsl_rl/tt_flat_motion_debug/2026-04-23_14-09-22_platform_001_g1_flat_start_n8192_it10000"

## visualizing platform_001 on flat
# uv run python -m mjlab.scripts.play \
#   "${TASK}" \
#   --agent zero \
#   --viewer "${VIEWER}" \
#   --motion-file "${MOTION_FILE}" \
#   --no-terminations True \
#   "$@"

## visualizing trained checkpoint on flat
uv run python -m mjlab.scripts.play \
  "${TASK}" \
  --agent "${AGENT}" \
  --viewer "${VIEWER}" \
  --motion-file "${MOTION_FILE}" \
  --checkpoint-file "${CHECKPOINT_DIR}/model_1000.pt" \
  --no-terminations True \
  "$@"
