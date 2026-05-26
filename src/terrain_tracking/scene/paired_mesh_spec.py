from __future__ import annotations

from collections.abc import Callable
from typing import cast

import mujoco
import numpy as np
import trimesh

from terrain_tracking.runtime.pair_manifest import PairManifest


def compute_env_origins_grid(num_envs: int, env_spacing: float) -> np.ndarray:
  if num_envs <= 0:
    raise ValueError("num_envs must be positive")
  env_origins = np.zeros((num_envs, 3), dtype=np.float32)
  num_rows = int(np.ceil(num_envs / int(np.sqrt(num_envs))))
  num_cols = int(np.ceil(num_envs / num_rows))
  ii, jj = np.meshgrid(np.arange(num_rows), np.arange(num_cols), indexing="ij")
  env_origins[:, 0] = -(ii.reshape(-1)[:num_envs] - (num_rows - 1) / 2) * env_spacing
  env_origins[:, 1] = (jj.reshape(-1)[:num_envs] - (num_cols - 1) / 2) * env_spacing
  return env_origins


def _as_trimesh(mesh_or_scene: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
  if isinstance(mesh_or_scene, trimesh.Trimesh):
    return mesh_or_scene
  if not mesh_or_scene.geometry:
    raise ValueError("Terrain scene does not contain any geometry")
  return trimesh.util.concatenate(tuple(mesh_or_scene.geometry.values()))


def _rotation_matrix_from_xyzw(quat_xyzw: tuple[float, float, float, float]) -> np.ndarray:
  x, y, z, w = quat_xyzw
  norm = np.linalg.norm([x, y, z, w])
  if norm == 0.0:
    raise ValueError("terrain_quat_xyzw must be non-zero")
  x, y, z, w = np.asarray([x, y, z, w], dtype=np.float64) / norm
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ],
    dtype=np.float32,
  )


def _load_mesh_arrays(
  manifest: PairManifest,
) -> tuple[np.ndarray, np.ndarray]:
  mesh_or_scene = cast(
    trimesh.Trimesh | trimesh.Scene,
    trimesh.load(manifest.terrain_file, force="mesh"),
  )
  mesh = _as_trimesh(mesh_or_scene)

  vertices = np.asarray(mesh.vertices, dtype=np.float32).copy()
  faces = np.asarray(mesh.faces, dtype=np.int32).copy()

  vertices *= np.asarray(manifest.terrain_scale, dtype=np.float32)
  vertices = vertices @ _rotation_matrix_from_xyzw(manifest.terrain_quat_xyzw).T
  vertices += np.asarray(manifest.terrain_translation, dtype=np.float32)

  return vertices, faces


def make_paired_mesh_spec_fn(
  manifest: PairManifest,
  *,
  num_envs: int,
  env_spacing: float,
  mesh_name: str = "paired_terrain_mesh",
) -> Callable[[mujoco.MjSpec], None]:
  vertices, faces = _load_mesh_arrays(manifest)
  env_origins = compute_env_origins_grid(num_envs=num_envs, env_spacing=env_spacing)

  def spec_fn(spec: mujoco.MjSpec) -> None:
    spec.add_mesh(
      name=mesh_name,
      uservert=vertices.reshape(-1).tolist(),
      userface=faces.reshape(-1).tolist(),
    )
    for env_id, env_origin in enumerate(env_origins):
      body = spec.worldbody.add_body(
        name=f"paired_terrain_{env_id}",
        pos=env_origin.tolist(),
      )
      body.add_geom(
        name=f"paired_terrain_{env_id}",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh_name,
      )

  return spec_fn
