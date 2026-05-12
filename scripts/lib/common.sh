#!/usr/bin/env bash

TT_SCRIPTS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)
TT_REPO_ROOT=$(cd -- "${TT_SCRIPTS_DIR}/.." &>/dev/null && pwd)

tt_cd_repo_root() {
  cd "${TT_REPO_ROOT}"
}

tt_source_wandb_env() {
  local wandb_env_file="${1}"

  if [[ -f "${wandb_env_file}" ]]; then
    # shellcheck disable=SC1090
    source "${wandb_env_file}"
  fi
}

tt_require_file() {
  local file_path="${1}"
  local description="${2}"

  if [[ ! -f "${file_path}" ]]; then
    echo "${description} not found: ${file_path}" >&2
    exit 1
  fi
}

tt_export_cuda() {
  local cuda_visible_devices="${1}"
  local gpu_ids="${2}"

  export CUDA_VISIBLE_DEVICES="${cuda_visible_devices}"
  echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; mjlab --gpu-ids ${gpu_ids}"
}
