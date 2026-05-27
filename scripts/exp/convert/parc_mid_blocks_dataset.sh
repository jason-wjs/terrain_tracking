#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "${SCRIPT_DIR}/../../lib/common.sh"

CONVERT_KIND="${CONVERT_KIND:-parc_pair_dataset}"
PARC_ROOT="${PARC_ROOT:-/home/humanoid/Downloads/Data/parc_initial_aug_g1}"
OUTPUT_FILE="${OUTPUT_FILE:-${PARC_ROOT}/pair_dataset_mid_blocks.jsonl}"
PAIR_DIR_NAME_PREFIX="${PAIR_DIR_NAME_PREFIX:-mid_blocks}"
INCLUDE_PATH_PARTS="${INCLUDE_PATH_PARTS:-mj/mid_climbing}"

tt_convert_exp "$@"
