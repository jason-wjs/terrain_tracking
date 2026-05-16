from __future__ import annotations

import hashlib
import importlib
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from terrain_tracking.runtime.pair_manifest import PairManifest
from terrain_tracking.scene.paired_mesh_spec import (
  _load_mesh_arrays,
  compute_env_origins_grid,
)

_COACD_CACHE_VERSION = 1
_COACD_PARTS_CACHE: dict[tuple[object, ...], list[tuple[np.ndarray, np.ndarray]]] = {}


@dataclass(frozen=True)
class CoacdCollisionOptions:
  threshold: float = 0.05
  max_convex_hull: int = -1
  preprocess_mode: str = "auto"
  preprocess_resolution: int = 50
  resolution: int = 2000
  mcts_nodes: int = 20
  mcts_iterations: int = 150
  mcts_max_depth: int = 3
  seed: int = 0
  log_level: str = "off"
  pca: bool = False
  merge: bool = False
  decimate: bool = False
  max_ch_vertex: int = 256
  extrude: bool = False
  extrude_margin: float = 0.01
  apx_mode: str = "ch"
  use_disk_cache: bool = True
  cache_dir: str | Path | None = None
  geom_margin: float = 0.0
  z_offset: float = 0.0
  visualize_collision_hulls: bool = False
  source_visual_group: int = 2
  collision_geom_group: int = 3
  collision_geom_rgba: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


def _import_coacd():
  return importlib.import_module("coacd")


def _coacd_run_kwargs(options: CoacdCollisionOptions) -> dict[str, object]:
  return {
    "threshold": float(options.threshold),
    "max_convex_hull": int(options.max_convex_hull),
    "preprocess_mode": str(options.preprocess_mode),
    "preprocess_resolution": int(options.preprocess_resolution),
    "resolution": int(options.resolution),
    "mcts_nodes": int(options.mcts_nodes),
    "mcts_iterations": int(options.mcts_iterations),
    "mcts_max_depth": int(options.mcts_max_depth),
    "pca": bool(options.pca),
    "merge": bool(options.merge),
    "decimate": bool(options.decimate),
    "max_ch_vertex": int(options.max_ch_vertex),
    "extrude": bool(options.extrude),
    "extrude_margin": float(options.extrude_margin),
    "apx_mode": str(options.apx_mode),
    "seed": int(options.seed),
  }


def _cache_parameter_tuple(options: CoacdCollisionOptions) -> tuple[object, ...]:
  return (
    float(options.threshold),
    int(options.max_convex_hull),
    str(options.preprocess_mode),
    int(options.preprocess_resolution),
    int(options.resolution),
    int(options.mcts_nodes),
    int(options.mcts_iterations),
    int(options.mcts_max_depth),
    int(options.seed),
    bool(options.pca),
    bool(options.merge),
    bool(options.decimate),
    int(options.max_ch_vertex),
    bool(options.extrude),
    float(options.extrude_margin),
    str(options.apx_mode),
  )


def _coacd_cache_key(
  pair: PairManifest,
  options: CoacdCollisionOptions,
) -> tuple[object, ...]:
  terrain_path = pair.terrain_file.resolve()
  stat_info = terrain_path.stat()
  return (
    _COACD_CACHE_VERSION,
    str(terrain_path),
    int(stat_info.st_size),
    int(stat_info.st_mtime_ns),
    tuple(float(v) for v in pair.terrain_scale),
    tuple(float(v) for v in pair.terrain_quat_xyzw),
    tuple(float(v) for v in pair.terrain_translation),
    *_cache_parameter_tuple(options),
  )


def _resolve_cache_dir(options: CoacdCollisionOptions) -> Path:
  if options.cache_dir is not None:
    return Path(options.cache_dir).expanduser().resolve()
  return (Path.cwd() / ".cache" / "terrain_tracking" / "coacd").resolve()


def _coacd_cache_path(
  pair: PairManifest,
  options: CoacdCollisionOptions,
  cache_key: tuple[object, ...],
) -> Path:
  key_hash = hashlib.sha1(repr(cache_key).encode("utf-8")).hexdigest()[:16]
  terrain_stem = pair.terrain_file.stem
  return _resolve_cache_dir(options) / f"{terrain_stem}.{key_hash}.npz"


def _load_coacd_parts_from_disk(cache_path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
  with np.load(cache_path, allow_pickle=False) as cache:
    if "num_parts" not in cache:
      raise ValueError(f"Invalid CoACD cache file (missing num_parts): {cache_path}")
    num_parts = int(np.asarray(cache["num_parts"]).reshape(-1)[0])
    parts: list[tuple[np.ndarray, np.ndarray]] = []
    for part_idx in range(num_parts):
      verts_key = f"verts_{part_idx}"
      faces_key = f"faces_{part_idx}"
      if verts_key not in cache or faces_key not in cache:
        raise ValueError(
          "Invalid CoACD cache file "
          f"(missing part arrays): {cache_path}, part_idx={part_idx}"
        )
      parts.append(
        (
          np.asarray(cache[verts_key], dtype=np.float32),
          np.asarray(cache[faces_key], dtype=np.int32),
        )
      )
  return parts


def _save_coacd_parts_to_disk(
  cache_path: Path,
  parts: list[tuple[np.ndarray, np.ndarray]],
) -> None:
  cache_path.parent.mkdir(parents=True, exist_ok=True)
  payload: dict[str, np.ndarray] = {
    "num_parts": np.asarray([len(parts)], dtype=np.int32),
  }
  for part_idx, (verts, faces) in enumerate(parts):
    payload[f"verts_{part_idx}"] = np.asarray(verts, dtype=np.float32)
    payload[f"faces_{part_idx}"] = np.asarray(faces, dtype=np.int32)

  tmp_cache_path = cache_path.with_name(f"{cache_path.name}.{uuid.uuid4().hex}.tmp.npz")
  np.savez_compressed(str(tmp_cache_path), **payload)
  os.replace(tmp_cache_path, cache_path)


def _load_transformed_trimesh(pair: PairManifest) -> trimesh.Trimesh:
  vertices, faces = _load_mesh_arrays(pair)
  mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
  mesh.remove_unreferenced_vertices()
  trimesh.repair.fix_winding(mesh)
  trimesh.repair.fix_normals(mesh, multibody=True)
  return mesh


def _sanitize_coacd_parts(
  parts: list[tuple[np.ndarray, np.ndarray]],
  *,
  terrain_tag: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
  sanitized: list[tuple[np.ndarray, np.ndarray]] = []
  skipped_parts = 0
  total_parts = len(parts)

  for part_verts, part_faces in parts:
    verts = np.asarray(part_verts, dtype=np.float64)
    faces = np.asarray(part_faces, dtype=np.int64)

    if verts.ndim != 2 or verts.shape[1] != 3:
      skipped_parts += 1
      continue
    if faces.ndim != 2 or faces.shape[1] != 3:
      skipped_parts += 1
      continue
    if verts.shape[0] < 4 or faces.shape[0] < 4:
      skipped_parts += 1
      continue

    valid_faces = np.logical_and(faces >= 0, faces < verts.shape[0]).all(axis=1)
    if not np.any(valid_faces):
      skipped_parts += 1
      continue
    faces = faces[valid_faces]

    used_vertex_ids = np.unique(faces.reshape(-1))
    if used_vertex_ids.size < 4:
      skipped_parts += 1
      continue

    remap = np.full(verts.shape[0], -1, dtype=np.int64)
    remap[used_vertex_ids] = np.arange(used_vertex_ids.size, dtype=np.int64)
    verts = verts[used_vertex_ids]
    faces = remap[faces]

    sorted_faces = np.sort(faces, axis=1)
    _, unique_face_indices = np.unique(sorted_faces, axis=0, return_index=True)
    faces = faces[np.sort(unique_face_indices)]
    if faces.shape[0] < 4:
      skipped_parts += 1
      continue

    centered = verts - np.mean(verts, axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    if singular_values.shape[0] < 3:
      skipped_parts += 1
      continue
    largest_sv = max(float(singular_values[0]), 1.0e-12)
    smallest_sv = float(singular_values[-1])
    if smallest_sv / largest_sv < 1.0e-8:
      skipped_parts += 1
      continue

    extents = np.ptp(verts, axis=0)
    if float(np.min(extents)) < 1.0e-7:
      skipped_parts += 1
      continue

    sanitized.append((verts.astype(np.float32), faces.astype(np.int32)))

  if not sanitized:
    raise ValueError(
      "CoACD produced no valid 3D hulls for "
      f"{terrain_tag}; try increasing collision quality or checking the terrain mesh."
    )
  if skipped_parts > 0:
    print(f"[terrain_tracking] Skipped {skipped_parts}/{total_parts} degenerate CoACD hull(s).")
  return sanitized


def _run_coacd_decomposition(
  mesh: trimesh.Trimesh,
  options: CoacdCollisionOptions,
) -> list[tuple[np.ndarray, np.ndarray]]:
  try:
    coacd = _import_coacd()
  except ImportError as exc:
    raise RuntimeError(
      "collision_backend='coacd' requires the coacd Python package. "
      "Install project dependencies with uv sync or add coacd to the environment."
    ) from exc

  if hasattr(coacd, "set_log_level"):
    coacd.set_log_level(str(options.log_level))

  coacd_mesh = coacd.Mesh(
    vertices=np.asarray(mesh.vertices, dtype=np.float64),
    indices=np.asarray(mesh.faces, dtype=np.int32),
  )
  raw_parts = coacd.run_coacd(coacd_mesh, **_coacd_run_kwargs(options))
  parts = [
    (np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32))
    for verts, faces in raw_parts
  ]
  return _sanitize_coacd_parts(parts, terrain_tag=str(mesh.metadata.get("file_name", "terrain")))


def _load_or_compute_coacd_parts(
  pair: PairManifest,
  options: CoacdCollisionOptions,
) -> list[tuple[np.ndarray, np.ndarray]]:
  cache_key = _coacd_cache_key(pair, options)
  cached_parts = _COACD_PARTS_CACHE.get(cache_key)
  if cached_parts is not None:
    return cached_parts

  cache_path = _coacd_cache_path(pair, options, cache_key)
  if options.use_disk_cache and cache_path.exists():
    parts = _sanitize_coacd_parts(
      _load_coacd_parts_from_disk(cache_path),
      terrain_tag=f"{pair.terrain_file} (disk cache)",
    )
    _COACD_PARTS_CACHE[cache_key] = parts
    return parts

  mesh = _load_transformed_trimesh(pair)
  mesh.metadata["file_name"] = str(pair.terrain_file)
  parts = _run_coacd_decomposition(mesh, options)
  if options.use_disk_cache:
    _save_coacd_parts_to_disk(cache_path, parts)
  _COACD_PARTS_CACHE[cache_key] = parts
  return parts


def _validate_geom_group(value: int, *, name: str) -> int:
  value = int(value)
  if not 0 <= value <= 5:
    raise ValueError(f"{name} must be in [0, 5], got {value}")
  return value


def make_coacd_mesh_spec_fn(
  pair: PairManifest,
  *,
  num_envs: int,
  env_spacing: float,
  options: CoacdCollisionOptions | None = None,
  visual_mesh_name: str = "paired_terrain_visual_mesh",
  hull_mesh_prefix: str = "paired_terrain_coacd_hull",
) -> Callable[[mujoco.MjSpec], None]:
  options = options if options is not None else CoacdCollisionOptions()
  vertices, faces = _load_mesh_arrays(pair)
  parts = _load_or_compute_coacd_parts(pair, options)
  env_origins = compute_env_origins_grid(num_envs=num_envs, env_spacing=env_spacing)

  source_group = _validate_geom_group(options.source_visual_group, name="source_visual_group")
  hull_group = _validate_geom_group(options.collision_geom_group, name="collision_geom_group")
  if len(options.collision_geom_rgba) != 4:
    raise ValueError("collision_geom_rgba must contain four values")

  def spec_fn(spec: mujoco.MjSpec) -> None:
    spec.add_mesh(
      name=visual_mesh_name,
      uservert=np.asarray(vertices, dtype=np.float32).reshape(-1).tolist(),
      userface=np.asarray(faces, dtype=np.int32).reshape(-1).tolist(),
    )
    hull_mesh_names: list[str] = []
    for part_idx, (part_verts, part_faces) in enumerate(parts):
      hull_mesh_name = f"{hull_mesh_prefix}_{part_idx}"
      hull_mesh_names.append(hull_mesh_name)
      spec.add_mesh(
        name=hull_mesh_name,
        uservert=np.asarray(part_verts, dtype=np.float32).reshape(-1).tolist(),
        userface=np.asarray(part_faces, dtype=np.int32).reshape(-1).tolist(),
      )

    for env_id, env_origin in enumerate(env_origins):
      body = spec.worldbody.add_body(
        name=f"paired_terrain_{env_id}",
        pos=env_origin.tolist(),
      )
      visual_geom = body.add_geom(
        name=f"paired_terrain_{env_id}_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=visual_mesh_name,
        contype=0,
        conaffinity=0,
      )
      if options.visualize_collision_hulls:
        visual_geom.group = 4
        visual_geom.rgba[:] = (0.0, 0.0, 0.0, 0.0)
      else:
        visual_geom.group = source_group

      for part_idx, hull_mesh_name in enumerate(hull_mesh_names):
        hull_geom = body.add_geom(
          name=f"paired_terrain_{env_id}_coacd_{part_idx}",
          type=mujoco.mjtGeom.mjGEOM_MESH,
          meshname=hull_mesh_name,
          pos=(0.0, 0.0, float(options.z_offset)),
          contype=1,
          conaffinity=1,
        )
        hull_geom.group = 2 if options.visualize_collision_hulls else hull_group
        if not options.visualize_collision_hulls:
          hull_geom.rgba[:] = tuple(float(v) for v in options.collision_geom_rgba)
        hull_geom.margin = float(options.geom_margin)
        hull_geom.gap = 0.0

  return spec_fn
