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
# uv run python -m terrain_tracking.convert_pair \
#   --motion-file /tmp/tt_converted/mid_blocks_004_dm/motion.npz \
#   --terrain-file /tmp/parc_process_workspace/workspace/mid_blocks_004_dm/multi_boxes.obj \
#   --terrain-collision-file /tmp/parc_process_workspace/workspace/mid_blocks_004_dm/terrain_collision.json \
#   --terrain-visual-file /tmp/parc_process_workspace/workspace/mid_blocks_004_dm/multi_boxes.obj \
#   --output-dir "${OUTPUT_ROOT}" \
#   --sample-name mid_blocks_004_dm \
#   --terrain-translation 0 0 0 \
#   --terrain-quat-xyzw 0 0 0 1 \
#   "$@"

## beyond_dash_vault_001_aug001_dm
# uv run python -m terrain_tracking.convert_pair \
#   --motion-file /tmp/tt_converted/beyond_dash_vault_001_aug001_dm/motion.npz \
#   --terrain-file /tmp/parc_process_workspace/workspace/beyond_dash_vault_001_aug001_dm/multi_boxes.obj \
#   --terrain-collision-file /tmp/parc_process_workspace/workspace/beyond_dash_vault_001_aug001_dm/terrain_collision.json \
#   --terrain-visual-file /tmp/parc_process_workspace/workspace/beyond_dash_vault_001_aug001_dm/multi_boxes.obj \
#   --output-dir "${OUTPUT_ROOT}" \
#   --sample-name beyond_dash_vault_001_aug001_dm \
#   --terrain-translation 0 0 0 \
#   --terrain-quat-xyzw 0 0 0 1 \
#   "$@"

## climbing_up_down_terrain_001_aug001_dm_aug2
# uv run python -m terrain_tracking.convert_pair \
#   --motion-file /tmp/tt_converted/climbing_up_down_terrain_001_aug001_dm_aug2/motion.npz \
#   --terrain-file /tmp/parc_process_workspace/workspace/climbing_up_down_terrain_001_aug001_dm_aug2/multi_boxes.obj \
#   --terrain-collision-file /tmp/parc_process_workspace/workspace/climbing_up_down_terrain_001_aug001_dm_aug2/terrain_collision.json \
#   --terrain-visual-file /tmp/parc_process_workspace/workspace/climbing_up_down_terrain_001_aug001_dm_aug2/multi_boxes.obj \
#   --output-dir "${OUTPUT_ROOT}" \
#   --sample-name climbing_up_down_terrain_001_aug001_dm_aug2 \
#   --terrain-translation 0 0 0 \
#   --terrain-quat-xyzw 0 0 0 1 \
#   "$@"

## run_jump_gap_001_dm_dm_aug0
# uv run python -m terrain_tracking.convert_pair \
#   --motion-file /tmp/tt_converted/run_jump_gap_001_dm_dm_aug0/motion.npz \
#   --terrain-file /tmp/parc_process_workspace/workspace/run_jump_gap_001_dm_dm_aug0/multi_boxes.obj \
#   --terrain-collision-file /tmp/parc_process_workspace/workspace/run_jump_gap_001_dm_dm_aug0/terrain_collision.json \
#   --terrain-visual-file /tmp/parc_process_workspace/workspace/run_jump_gap_001_dm_dm_aug0/multi_boxes.obj \
#   --output-dir "${OUTPUT_ROOT}" \
#   --sample-name run_jump_gap_001_dm_dm_aug0 \
#   --terrain-translation 0 0 0 \
#   --terrain-quat-xyzw 0 0 0 1 \
#   "$@"

## run_jump_gap_001_dm_dm_aug0_flipped
# uv run python -m terrain_tracking.convert_pair \
#   --motion-file /tmp/tt_converted/run_jump_gap_001_dm_dm_aug0_flipped/motion.npz \
#   --terrain-file /tmp/parc_process_workspace/workspace/run_jump_gap_001_dm_dm_aug0_flipped/multi_boxes.obj \
#   --terrain-collision-file /tmp/parc_process_workspace/workspace/run_jump_gap_001_dm_dm_aug0_flipped/terrain_collision.json \
#   --terrain-visual-file /tmp/parc_process_workspace/workspace/run_jump_gap_001_dm_dm_aug0_flipped/multi_boxes.obj \
#   --output-dir "${OUTPUT_ROOT}" \
#   --sample-name run_jump_gap_001_dm_dm_aug0_flipped \
#   --terrain-translation 0 0 0 \
#   --terrain-quat-xyzw 0 0 0 1 \
#   "$@"

## beyond_stairs_001_before_opt_dm_aug0
# uv run python -m terrain_tracking.convert_pair \
#   --motion-file /tmp/tt_converted/beyond_stairs_001_before_opt_dm_aug0/motion.npz \
#   --terrain-file /tmp/parc_process_workspace/workspace/beyond_stairs_001_before_opt_dm_aug0/multi_boxes.obj \
#   --terrain-collision-file /tmp/parc_process_workspace/workspace/beyond_stairs_001_before_opt_dm_aug0/terrain_collision.json \
#   --terrain-visual-file /tmp/parc_process_workspace/workspace/beyond_stairs_001_before_opt_dm_aug0/multi_boxes.obj \
#   --output-dir "${OUTPUT_ROOT}" \
#   --sample-name beyond_stairs_001_before_opt_dm_aug0 \
#   --terrain-translation 0 0 0 \
#   --terrain-quat-xyzw 0 0 0 1 \
#   "$@"

## mdm_castle_stairs_dm.pkl
uv run python -m terrain_tracking.convert_pair \
  --motion-file /tmp/tt_converted/mdm_castle_stairs_dm/motion.npz \
  --terrain-file /tmp/parc_process_workspace/workspace/mdm_castle_stairs_dm/multi_boxes.obj \
  --terrain-collision-file /tmp/parc_process_workspace/workspace/mdm_castle_stairs_dm/terrain_collision.json \
  --terrain-visual-file /tmp/parc_process_workspace/workspace/mdm_castle_stairs_dm/multi_boxes.obj \
  --output-dir "${OUTPUT_ROOT}" \
  --sample-name mdm_castle_stairs_dm \
  --terrain-translation 0 0 0 \
  --terrain-quat-xyzw 0 0 0 1 \
  "$@"