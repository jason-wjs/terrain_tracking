#!/usr/bin/env bash
set -euo pipefail

PAIR_MANIFEST="${1:?usage: $0 /path/to/pair.json}"

uv run python - "${PAIR_MANIFEST}" <<'PY'
import sys

import mujoco
import torch

from terrain_tracking.tasks.blind_terrain_tracking.scripts.common import build_paired_env

pair_manifest = sys.argv[1]
env, _agent_cfg = build_paired_env(
  "TT-Tracking-TerrainBlind-Unitree-G1",
  pair_manifest,
  play=True,
  device="cpu",
  num_envs=1,
  no_terminations=True,
)
try:
  model = env.unwrapped.sim.mj_model
  env.reset()
  terrain_geoms = []
  for geom_id in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
    if name.startswith("paired_terrain_"):
      terrain_geoms.append((name, int(model.geom_type[geom_id])))
  print("terrain_geoms", terrain_geoms)
  assert any(
    geom_type == mujoco.mjtGeom.mjGEOM_HFIELD
    for _, geom_type in terrain_geoms
  ), "pair did not compile a MuJoCo hfield terrain geom"

  action_dim = env.unwrapped.single_action_space.shape[0]
  action = torch.zeros((1, action_dim), device=env.device)
  for _ in range(5):
    env.step(action)

  data = env.unwrapped.sim.mj_data
  min_contact_dist = 0.0
  for contact_id in range(data.ncon):
    min_contact_dist = min(min_contact_dist, float(data.contact[contact_id].dist))
  print("ncon", data.ncon, "min_contact_dist", min_contact_dist)
finally:
  env.close()
PY
