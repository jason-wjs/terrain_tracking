#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)
TASK="${TASK:-TT-Tracking-TerrainBlind-Unitree-G1}"

cd "${REPO_ROOT}"

## platform_001
uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.train \
  --task "${TASK}" \
  --agent.experiment-name "tt_single_pair_platform_001" \
  --agent.run-name "platform_001_g1_blind_NoPlaneCollision_start_n8192_it10000" \
  --collision-backend primitive_boxes \
  --pair-manifest /tmp/tt_converted/platform_001/pair.json \
  --env.scene.num-envs "8192" \
  --env.scene.env-spacing 12.0 \
  --agent.max-iterations "10000" \
  "$@"

## mid_blocks_004_dm
# uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.train \
#   --task "${TASK}" \
#   --agent.experiment-name "tt_single_pair_mid_blocks_004_dm" \
#   --agent.run-name "mid_blocks_004_dm_g1_blind_n32768_it10000" \
#   --pair-manifest /tmp/tt_converted/mid_blocks_004_dm/pair.json \
#   --collision-backend primitive_boxes \
#   --env.scene.num-envs "32768" \
#   --env.scene.env-spacing 12.0 \
#   --agent.max-iterations "10000" \
#   "$@"
