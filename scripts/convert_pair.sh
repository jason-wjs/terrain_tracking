#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)
cd "${REPO_ROOT}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/tt_converted}"

# platform_001
# pair manifest: /tmp/tt_converted/platform_001/pair.json
# uv run python -m terrain_tracking.convert_pair \
#   --motion-file /tmp/tt_converted/platform_001/motion.npz \
#   --terrain-file /tmp/parc_process_workspace/workspace/platform_001/multi_boxes.obj \
#   --terrain-collision-file /tmp/parc_process_workspace/workspace/platform_001/terrain_collision.json \
#   --terrain-visual-file /tmp/parc_process_workspace/workspace/platform_001/multi_boxes.obj \
#   --output-dir "${OUTPUT_ROOT}" \
#   --sample-name platform_001 \
#   --terrain-translation 0 0 0 \
#   --terrain-quat-xyzw 0 0 0 1 \
#   "$@"

# mid_blocks_004_dm
# pair manifest: /tmp/tt_converted/mid_blocks_004_dm/pair.json
uv run python -m terrain_tracking.convert_pair \
  --motion-file /tmp/tt_converted/mid_blocks_004_dm/motion.npz \
  --terrain-file /tmp/parc_process_workspace/workspace/mid_blocks_004_dm/multi_boxes.obj \
  --terrain-collision-file /tmp/parc_process_workspace/workspace/mid_blocks_004_dm/terrain_collision.json \
  --terrain-visual-file /tmp/parc_process_workspace/workspace/mid_blocks_004_dm/multi_boxes.obj \
  --output-dir "${OUTPUT_ROOT}" \
  --sample-name mid_blocks_004_dm \
  --terrain-translation 0 0 0 \
  --terrain-quat-xyzw 0 0 0 1 \
  "$@"
