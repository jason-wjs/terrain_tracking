#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "${SCRIPT_DIR}/../../lib/common.sh"

OMNIRETARGET_ROOT="${OMNIRETARGET_ROOT:-/path/to/OmniRetarget_Dataset}"
CONVERT_KIND="${CONVERT_KIND:-omniretarget_robot_terrain}"
MOTION_FILE="${MOTION_FILE:-${OMNIRETARGET_ROOT}/robot-terrain/climb_00_z_scale_1.0.npz}"
TERRAIN_ROOT="${TERRAIN_ROOT:-${OMNIRETARGET_ROOT}/models/terrain}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/tt_converted_omniretarget}"
SAMPLE_NAME="${SAMPLE_NAME:-climb_00_z_scale_1.0}"

tt_convert_exp "$@"
