#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)
cd "${REPO_ROOT}"

PAIR_MANIFEST="${1:-/tmp/tt_converted/platform_001/pair.json}"
BACKEND="${2:-primitive_boxes}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-terrain-tracking}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/warp-cache-terrain-tracking}"
mkdir -p "${MPLCONFIGDIR}" "${WARP_CACHE_PATH}"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  "${REPO_ROOT}/.venv/bin/python" -m terrain_tracking.tasks.blind_terrain_tracking.scripts.debug_zero_action \
    --pair-manifest "${PAIR_MANIFEST}" \
    --collision-backend "${BACKEND}" \
    --num-envs 64 \
    --steps 12
else
  uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.debug_zero_action \
    --pair-manifest "${PAIR_MANIFEST}" \
    --collision-backend "${BACKEND}" \
    --num-envs 64 \
    --steps 12
fi
