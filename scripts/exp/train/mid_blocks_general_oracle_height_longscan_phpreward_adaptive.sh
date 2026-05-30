#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
source "${SCRIPT_DIR}/../../lib/common.sh"

## general oracle height long-scan PHP reward over all PARC mid_blocks pairs.
TASK="${TASK:-TT-Tracking-TerrainOracleHeightLongScanPhpRewardGeneral-Unitree-G1}"
PAIR_DATASET="${PAIR_DATASET:-/home/humanoid/Downloads/Data/parc_initial_aug_g1/pair_dataset_mid_blocks.jsonl}"
DATASET_VALIDATE="${DATASET_VALIDATE:-fast}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-tt_general_pair_dataset_mid_blocks}"
RUN_NAME="${RUN_NAME:-mid_blocks_general_g1_oracle_height_longscan_phpreward_n16384_adaptive}"
SAMPLING_MODE="${SAMPLING_MODE:-adaptive}"
PAIR_SAMPLER_MODE="${PAIR_SAMPLER_MODE:-independent}"
NUM_ENVS="${NUM_ENVS:-16384}"
MAX_ITERATIONS="${MAX_ITERATIONS:-50000}"

tt_train_general_pair_dataset_exp "$@"
