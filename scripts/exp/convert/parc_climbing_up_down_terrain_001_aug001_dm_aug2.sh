#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "${SCRIPT_DIR}/../../lib/common.sh"

CONVERT_KIND="${CONVERT_KIND:-pair}"
MOTION_FILE="${MOTION_FILE:-/tmp/tt_converted/climbing_up_down_terrain_001_aug001_dm_aug2/motion.npz}"
TERRAIN_FILE="${TERRAIN_FILE:-/tmp/parc_process_workspace/workspace/climbing_up_down_terrain_001_aug001_dm_aug2/multi_boxes.obj}"
TERRAIN_COLLISION_FILE="${TERRAIN_COLLISION_FILE:-/tmp/parc_process_workspace/workspace/climbing_up_down_terrain_001_aug001_dm_aug2/terrain_collision.json}"
TERRAIN_VISUAL_FILE="${TERRAIN_VISUAL_FILE:-/tmp/parc_process_workspace/workspace/climbing_up_down_terrain_001_aug001_dm_aug2/multi_boxes.obj}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/tt_converted}"
SAMPLE_NAME="${SAMPLE_NAME:-climbing_up_down_terrain_001_aug001_dm_aug2}"

tt_convert_exp "$@"
