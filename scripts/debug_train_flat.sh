#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)
cd "${REPO_ROOT}"

TASK="Mjlab-Tracking-Flat-Unitree-G1"

uv run python -m mjlab.scripts.train \
  "${TASK}" \
  --agent.experiment-name "tt_flat_motion_debug" \
  --agent.run-name "platform_001_g1_flat_start_n8192_it10000" \
  --env.scene.num-envs "8192" \
  --agent.max-iterations "10000" \
  --env.commands.motion.motion-file "/tmp/tt_converted/platform_001/motion.npz" \
  --env.commands.motion.sampling-mode "start" \
  "$@"
