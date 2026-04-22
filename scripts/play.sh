#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)
cd "${REPO_ROOT}"

TASK="${TASK:-TT-Tracking-TerrainBlind-Unitree-G1}"
AGENT="${AGENT:-trained}"
VIEWER="${VIEWER:-viser}"
DATA_DIR="/home/humanoid/Downloads/Data/G1_retargeted/lafan1_npz"
CHECKPOINT_DIR="/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/logs/rsl_rl/tt_single_pair_platform_001/2026-04-22_11-38-09_platform_001_g1_blind_n16384_it10000/"

## visualizing platform_001
uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.play \
  --task "${TASK}" \
  --agent zero \
  --viewer "${VIEWER}" \
  --pair-manifest /tmp/tt_converted/platform_001/pair.json \
  "$@"


## platform_001
# pair manifest: /tmp/tt_converted/platform_001/pair.json
# uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.play \
#   --task "${TASK}" \
#   --agent "${AGENT}" \
#   --viewer "${VIEWER}" \
#   --pair-manifest /tmp/tt_converted/platform_001/pair.json \
#   --checkpoint-file "${CHECKPOINT_DIR}/model_8000.pt" \
#   "$@"

## mid_blocks_004_dm
# pair manifest: /tmp/tt_converted/mid_blocks_004_dm/pair.json
# uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.play \
#   --task "${TASK}" \
#   --agent "${AGENT}" \
#   --viewer "${VIEWER}" \
#   --pair-manifest /tmp/tt_converted/mid_blocks_004_dm/pair.json \
#   "$@"




