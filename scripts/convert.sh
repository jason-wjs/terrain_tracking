#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "${SCRIPT_DIR}/lib/common.sh"

CONVERT_KIND="${CONVERT_KIND:-pair}"

tt_convert_exp "$@"
