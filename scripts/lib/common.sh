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

tt_require_dir() {
  local dir_path="${1}"
  local description="${2}"

  if [[ ! -d "${dir_path}" ]]; then
    echo "${description} not found: ${dir_path}" >&2
    exit 1
  fi
}

tt_require_value() {
  local value="${1}"
  local description="${2}"

  if [[ -z "${value}" ]]; then
    echo "${description} is required" >&2
    exit 1
  fi
}

tt_export_cuda() {
  local cuda_visible_devices="${1}"
  local gpu_ids="${2}"

  export CUDA_VISIBLE_DEVICES="${cuda_visible_devices}"
  echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; mjlab --gpu-ids ${gpu_ids}"
}

tt_latest_checkpoint_file() {
  local experiment_name="${1}"
  local run_name="${2}"
  local checkpoint_dir="${3:-}"

  if [[ -z "${checkpoint_dir}" ]]; then
    shopt -s nullglob
    local matching_run_dirs=("${TT_REPO_ROOT}/logs/rsl_rl/${experiment_name}"/*_"${run_name}")
    shopt -u nullglob

    if (( ${#matching_run_dirs[@]} == 0 )); then
      echo "No run directory found for run '${run_name}' under logs/rsl_rl/${experiment_name}." >&2
      echo "Train first via scripts/exp/train/<experiment>.sh or set CHECKPOINT_DIR/CHECKPOINT_FILE explicitly." >&2
      return 1
    fi

    local -a sorted_run_dirs
    IFS=$'\n' sorted_run_dirs=($(printf '%s\n' "${matching_run_dirs[@]}" | sort))
    unset IFS
    local last_run_idx=$((${#sorted_run_dirs[@]} - 1))
    checkpoint_dir="${sorted_run_dirs[${last_run_idx}]}"
  fi

  if [[ ! -d "${checkpoint_dir}" ]]; then
    echo "Checkpoint directory not found: ${checkpoint_dir}" >&2
    return 1
  fi

  shopt -s nullglob
  local matching_models=("${checkpoint_dir}"/model_*.pt)
  shopt -u nullglob

  if (( ${#matching_models[@]} == 0 )); then
    echo "No model_*.pt checkpoint found under '${checkpoint_dir}'." >&2
    echo "Set CHECKPOINT_FILE explicitly or wait for checkpoints to be saved." >&2
    return 1
  fi

  local -a sorted_models
  IFS=$'\n' sorted_models=($(printf '%s\n' "${matching_models[@]}" | sort -V))
  unset IFS
  local last_model_idx=$((${#sorted_models[@]} - 1))
  printf '%s\n' "${sorted_models[${last_model_idx}]}"
}

tt_train_exp() {
  local task="${TASK:-TT-Tracking-TerrainBlind-Unitree-G1}"
  local pair_manifest="${PAIR_MANIFEST:-/tmp/tt_converted/beyond_dash_vault_001_aug001_dm/pair.json}"
  local experiment_name="${EXPERIMENT_NAME:-tt_single_pair_beyond_dash_vault_001_aug001_dm}"
  local run_name="${RUN_NAME:-beyond_dash_vault_001_aug001_dm_g1_blind_primitive_boxes_n32768_it10000}"
  local collision_backend="${COLLISION_BACKEND:-primitive_boxes}"
  local sampling_mode="${SAMPLING_MODE:-start}"
  local num_envs="${NUM_ENVS:-32768}"
  local env_spacing="${ENV_SPACING:-12.0}"
  local max_iterations="${MAX_ITERATIONS:-100000}"
  local cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"
  local gpu_ids="${GPU_IDS:-[0]}"
  local wandb_env_file="${WANDB_ENV_FILE:-/data/junsong/.secrets/wandb.env}"

  tt_cd_repo_root
  tt_source_wandb_env "${wandb_env_file}"
  tt_require_file "${pair_manifest}" "Pair manifest"
  tt_export_cuda "${cuda_visible_devices}" "${gpu_ids}"

  uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.train \
    --task "${task}" \
    --agent.experiment-name "${experiment_name}" \
    --agent.run-name "${run_name}" \
    --collision-backend "${collision_backend}" \
    --env.commands.motion.sampling-mode "${sampling_mode}" \
    --pair-manifest "${pair_manifest}" \
    --env.scene.num-envs "${num_envs}" \
    --env.scene.env-spacing "${env_spacing}" \
    --agent.max-iterations "${max_iterations}" \
    --gpu-ids "${gpu_ids}" \
    "$@"
}

tt_play_exp() {
  local task="${TASK:-TT-Tracking-TerrainOracleTeacher-Unitree-G1}"
  local agent="${AGENT:-trained}"
  local viewer="${VIEWER:-viser}"
  local pair_manifest="${PAIR_MANIFEST:-/tmp/tt_converted/mid_blocks_004_dm/pair.json}"
  local experiment_name="${EXPERIMENT_NAME:-tt_single_pair_mid_blocks_004_dm}"
  local run_name="${RUN_NAME:-mid_blocks_004_dm_g1_oracle_teacher_n8192_adaptive}"
  local collision_backend="${COLLISION_BACKEND:-primitive_boxes}"
  local no_terminations="${NO_TERMINATIONS:-True}"
  local play_num_envs="${PLAY_NUM_ENVS:-}"
  local env_spacing="${ENV_SPACING:-}"
  local device="${DEVICE:-}"
  local checkpoint_dir="${CHECKPOINT_DIR:-}"
  local checkpoint_file="${CHECKPOINT_FILE:-}"

  tt_cd_repo_root
  tt_require_file "${pair_manifest}" "Pair manifest"

  if [[ "${agent}" != "zero" && "${agent}" != "random" ]]; then
    if [[ -z "${checkpoint_file}" ]]; then
      checkpoint_file=$(tt_latest_checkpoint_file "${experiment_name}" "${run_name}" "${checkpoint_dir}")
    fi
    tt_require_file "${checkpoint_file}" "Checkpoint file"
    echo "Using checkpoint: ${checkpoint_file}"
  fi

  local play_args=(
    --task "${task}"
    --agent "${agent}"
    --viewer "${viewer}"
    --pair-manifest "${pair_manifest}"
    --collision-backend "${collision_backend}"
    --no-terminations "${no_terminations}"
  )

  if [[ -n "${checkpoint_file}" ]]; then
    play_args+=(--checkpoint-file "${checkpoint_file}")
  fi

  if [[ -n "${play_num_envs}" ]]; then
    play_args+=(--num-envs "${play_num_envs}")
  fi

  if [[ -n "${env_spacing}" ]]; then
    play_args+=(--env-spacing "${env_spacing}")
  fi

  if [[ -n "${device}" ]]; then
    play_args+=(--device "${device}")
  fi

  uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.play \
    "${play_args[@]}" \
    "$@"
}

tt_convert_pair() {
  local motion_file="${MOTION_FILE:-}"
  local terrain_file="${TERRAIN_FILE:-}"
  local terrain_collision_file="${TERRAIN_COLLISION_FILE:-}"
  local terrain_visual_file="${TERRAIN_VISUAL_FILE:-}"
  local output_root="${OUTPUT_ROOT:-/tmp/tt_converted}"
  local sample_name="${SAMPLE_NAME:-}"
  local terrain_translation="${TERRAIN_TRANSLATION:-0 0 0}"
  local terrain_quat_xyzw="${TERRAIN_QUAT_XYZW:-0 0 0 1}"
  local terrain_scale="${TERRAIN_SCALE:-1 1 1}"

  tt_cd_repo_root
  tt_require_value "${motion_file}" "MOTION_FILE"
  tt_require_value "${terrain_file}" "TERRAIN_FILE"
  tt_require_file "${motion_file}" "Motion file"
  tt_require_file "${terrain_file}" "Terrain file"
  if [[ -n "${terrain_collision_file}" ]]; then
    tt_require_file "${terrain_collision_file}" "Terrain collision file"
  fi
  if [[ -n "${terrain_visual_file}" ]]; then
    tt_require_file "${terrain_visual_file}" "Terrain visual file"
  fi

  local -a terrain_translation_args
  local -a terrain_quat_xyzw_args
  local -a terrain_scale_args
  read -r -a terrain_translation_args <<< "${terrain_translation}"
  read -r -a terrain_quat_xyzw_args <<< "${terrain_quat_xyzw}"
  read -r -a terrain_scale_args <<< "${terrain_scale}"

  local -a convert_args=(
    --motion-file "${motion_file}"
    --terrain-file "${terrain_file}"
    --output-dir "${output_root}"
    --terrain-translation "${terrain_translation_args[@]}"
    --terrain-quat-xyzw "${terrain_quat_xyzw_args[@]}"
    --terrain-scale "${terrain_scale_args[@]}"
  )

  if [[ -n "${sample_name}" ]]; then
    convert_args+=(--sample-name "${sample_name}")
  fi
  if [[ -n "${terrain_collision_file}" ]]; then
    convert_args+=(--terrain-collision-file "${terrain_collision_file}")
  fi
  if [[ -n "${terrain_visual_file}" ]]; then
    convert_args+=(--terrain-visual-file "${terrain_visual_file}")
  fi

  uv run python -m terrain_tracking.convert_pair "${convert_args[@]}" "$@"
}

tt_convert_omniretarget_robot_terrain() {
  local motion_file="${MOTION_FILE:-}"
  local terrain_root="${TERRAIN_ROOT:-}"
  local output_root="${OUTPUT_ROOT:-/tmp/tt_converted_omniretarget}"
  local sample_name="${SAMPLE_NAME:-}"
  local output_fps="${OUTPUT_FPS:-50}"

  tt_cd_repo_root
  tt_require_value "${motion_file}" "MOTION_FILE"
  tt_require_value "${terrain_root}" "TERRAIN_ROOT"
  tt_require_file "${motion_file}" "OmniRetarget motion file"
  tt_require_dir "${terrain_root}" "OmniRetarget terrain root"

  local -a convert_args=(
    --motion-file "${motion_file}"
    --terrain-root "${terrain_root}"
    --output-dir "${output_root}"
    --output-fps "${output_fps}"
  )

  if [[ -n "${sample_name}" ]]; then
    convert_args+=(--sample-name "${sample_name}")
  fi

  uv run python -m terrain_tracking.convert_omniretarget_robot_terrain \
    "${convert_args[@]}" \
    "$@"
}

tt_convert_parc_pair_dataset() {
  local source="${SOURCE:-parc}"
  local parc_root="${PARC_ROOT:-/home/humanoid/Downloads/Data/parc_initial_aug_g1}"
  local output_file="${OUTPUT_FILE:-${parc_root}/pair_dataset.jsonl}"
  local pair_dir_name_prefix="${PAIR_DIR_NAME_PREFIX:-}"
  local include_path_parts="${INCLUDE_PATH_PARTS:-}"

  tt_cd_repo_root
  tt_require_dir "${parc_root}" "PARC root"

  local -a build_args=(
    --source "${source}"
    --root "${parc_root}"
    --output "${output_file}"
  )

  if [[ -n "${pair_dir_name_prefix}" ]]; then
    build_args+=(--pair-dir-name-prefix "${pair_dir_name_prefix}")
  fi

  if [[ -n "${include_path_parts}" ]]; then
    local -a include_path_part_args
    read -r -a include_path_part_args <<< "${include_path_parts}"
    for include_path_part in "${include_path_part_args[@]}"; do
      build_args+=(--include-path-part "${include_path_part}")
    done
  fi

  uv run python -m terrain_tracking.build_pair_dataset "${build_args[@]}" "$@"
}

tt_convert_exp() {
  local convert_kind="${CONVERT_KIND:-pair}"

  case "${convert_kind}" in
    pair)
      tt_convert_pair "$@"
      ;;
    omniretarget_robot_terrain)
      tt_convert_omniretarget_robot_terrain "$@"
      ;;
    parc_pair_dataset)
      tt_convert_parc_pair_dataset "$@"
      ;;
    *)
      echo "Unsupported CONVERT_KIND: ${convert_kind}" >&2
      echo "Expected one of: pair, omniretarget_robot_terrain, parc_pair_dataset" >&2
      exit 1
      ;;
  esac
}

tt_train_general_pair_dataset_exp() {
  local task="${TASK:-TT-Tracking-TerrainOracleTeacherGeneral-Unitree-G1}"
  local pair_dataset="${PAIR_DATASET:-/home/humanoid/Downloads/Data/parc_initial_aug_g1/pair_dataset.jsonl}"
  local dataset_validate="${DATASET_VALIDATE:-fast}"
  local experiment_name="${EXPERIMENT_NAME:-tt_general_pair_dataset}"
  local run_name="${RUN_NAME:-general_pair_dataset_g1_oracle_teacher_n8192_adaptive}"
  local sampling_mode="${SAMPLING_MODE:-adaptive}"
  local pair_sampler_mode="${PAIR_SAMPLER_MODE:-independent}"
  local num_envs="${NUM_ENVS:-8192}"
  local max_iterations="${MAX_ITERATIONS:-20000}"
  local max_pairs="${MAX_PAIRS:-}"
  local cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"
  local gpu_ids="${GPU_IDS:-[0]}"
  local wandb_env_file="${WANDB_ENV_FILE:-/data/junsong/.secrets/wandb.env}"

  tt_cd_repo_root
  tt_source_wandb_env "${wandb_env_file}"
  tt_require_file "${pair_dataset}" "Pair dataset"
  tt_export_cuda "${cuda_visible_devices}" "${gpu_ids}"

  local -a train_args=(
    --task "${task}"
    --agent.experiment-name "${experiment_name}"
    --agent.run-name "${run_name}"
    --pair-dataset "${pair_dataset}"
    --dataset-validate "${dataset_validate}"
    --pair-sampler-mode "${pair_sampler_mode}"
    --env.commands.motion.sampling-mode "${sampling_mode}"
    --env.scene.num-envs "${num_envs}"
    --agent.max-iterations "${max_iterations}"
    --gpu-ids "${gpu_ids}"
  )

  if [[ -n "${max_pairs}" ]]; then
    train_args+=(--max-pairs "${max_pairs}")
  fi

  uv run python -m terrain_tracking.tasks.general_terrain_tracking.scripts.train \
    "${train_args[@]}" \
    "$@"
}
