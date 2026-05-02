from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np

from terrain_tracking.runtime.terrain_collision import TerrainCollisionManifest
from terrain_tracking.scene.paired_mesh_spec import compute_env_origins_grid


def _cell_centered_to_vertex_heightfield(
  hf: np.ndarray,
  *,
  subdivisions: int,
) -> np.ndarray:
  if subdivisions <= 0:
    raise ValueError("subdivisions must be positive")
  nx, ny = hf.shape
  vertex_hf = np.full(
    (nx * subdivisions + 1, ny * subdivisions + 1),
    float(hf.min()),
    dtype=np.float32,
  )
  for i in range(nx):
    row_start = i * subdivisions
    row_end = (i + 1) * subdivisions + 1
    for j in range(ny):
      col_start = j * subdivisions
      col_end = (j + 1) * subdivisions + 1
      vertex_hf[row_start:row_end, col_start:col_end] = np.maximum(
        vertex_hf[row_start:row_end, col_start:col_end],
        hf[i, j],
      )
  return vertex_hf


def _load_heightfield(manifest: TerrainCollisionManifest) -> np.ndarray:
  hf = np.load(manifest.hf_file).astype(np.float32)
  if hf.ndim != 2:
    raise ValueError(f"heightfield must be 2D, got shape {hf.shape}")
  return hf * np.float32(manifest.height_scale)


def _validate_env_spacing(
  *,
  num_envs: int,
  env_spacing: float,
  size_x: float,
  size_y: float,
) -> None:
  if num_envs <= 1:
    return
  min_spacing = max(size_x, size_y)
  if env_spacing >= min_spacing:
    return
  raise ValueError(
    "env_spacing is too small for paired hfield terrain: "
    f"env_spacing={env_spacing:.6g}, required>={min_spacing:.6g} "
    f"(footprint_x={size_x:.6g}, footprint_y={size_y:.6g}). "
    "Increase --env-spacing or use one environment."
  )


def make_heightfield_spec_fn(
  manifest: TerrainCollisionManifest,
  *,
  num_envs: int,
  env_spacing: float,
  hfield_name: str = "paired_terrain_hfield",
  contact_slots_per_env: int = 128,
  cell_subdivisions: int = 4,
) -> Callable[[mujoco.MjSpec], None]:
  cell_hf = _load_heightfield(manifest)
  hf_xy = _cell_centered_to_vertex_heightfield(
    cell_hf,
    subdivisions=cell_subdivisions,
  )
  # MuJoCo hfield rows index local y and columns index local x.
  hf = hf_xy.T
  env_origins = compute_env_origins_grid(num_envs=num_envs, env_spacing=env_spacing)
  nx, ny = cell_hf.shape
  hfield_nrow, hfield_ncol = hf.shape
  size_x = float(nx * manifest.dx * manifest.xy_scale)
  size_y = float(ny * manifest.dx * manifest.xy_scale)
  _validate_env_spacing(
    num_envs=num_envs,
    env_spacing=env_spacing,
    size_x=size_x,
    size_y=size_y,
  )
  center_x = float(manifest.xy_scale * (manifest.min_point[0] + 0.5 * (nx - 1) * manifest.dx))
  center_y = float(manifest.xy_scale * (manifest.min_point[1] + 0.5 * (ny - 1) * manifest.dx))

  surface_min = float(hf.min())
  surface_max = float(hf.max())
  z_range = max(surface_max - surface_min, 1e-6)
  base_depth = max(surface_min - float(manifest.base_z), 1e-6)
  normalized_hf = ((hf - surface_min) / z_range).astype(np.float32)

  def spec_fn(spec: mujoco.MjSpec) -> None:
    min_contact_slots = int(num_envs * contact_slots_per_env)
    if spec.nconmax < min_contact_slots:
      spec.nconmax = min_contact_slots
    if spec.njmax < min_contact_slots:
      spec.njmax = min_contact_slots
    spec.add_hfield(
      name=hfield_name,
      nrow=hfield_nrow,
      ncol=hfield_ncol,
      size=[size_x / 2.0, size_y / 2.0, z_range, base_depth],
      userdata=normalized_hf.reshape(-1).tolist(),
    )
    for env_id, env_origin in enumerate(env_origins):
      body_pos = env_origin.astype(np.float64).copy()
      body_pos[0] += center_x
      body_pos[1] += center_y
      body_pos[2] += surface_min
      body = spec.worldbody.add_body(
        name=f"paired_terrain_{env_id}",
        pos=body_pos.tolist(),
      )
      body.add_geom(
        name=f"paired_terrain_{env_id}",
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=hfield_name,
        contype=1,
        conaffinity=1,
      )

  return spec_fn
