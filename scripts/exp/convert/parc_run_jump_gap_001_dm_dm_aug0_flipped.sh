#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "${SCRIPT_DIR}/../../lib/common.sh"

CONVERT_KIND="${CONVERT_KIND:-pair}"
MOTION_FILE="${MOTION_FILE:-/tmp/tt_converted/run_jump_gap_001_dm_dm_aug0_flipped/motion.npz}"
TERRAIN_FILE="${TERRAIN_FILE:-/tmp/parc_process_workspace/workspace/run_jump_gap_001_dm_dm_aug0_flipped/multi_boxes.obj}"
TERRAIN_COLLISION_FILE="${TERRAIN_COLLISION_FILE:-/tmp/parc_process_workspace/workspace/run_jump_gap_001_dm_dm_aug0_flipped/terrain_collision.json}"
TERRAIN_VISUAL_FILE="${TERRAIN_VISUAL_FILE:-/tmp/parc_process_workspace/workspace/run_jump_gap_001_dm_dm_aug0_flipped/multi_boxes.obj}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/tt_converted}"
SAMPLE_NAME="${SAMPLE_NAME:-run_jump_gap_001_dm_dm_aug0_flipped}"

tt_convert_exp "$@"
