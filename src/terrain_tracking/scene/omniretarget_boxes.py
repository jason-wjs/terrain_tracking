from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


@dataclass(frozen=True)
class OmniRetargetBox:
  name: str
  mesh_file: Path
  pos: tuple[float, float, float]
  size: tuple[float, float, float]
  yaw: float


def _parse_scale(value: str | None) -> np.ndarray:
  if value is None:
    return np.ones(3, dtype=np.float64)
  parts = value.split()
  if len(parts) != 3:
    raise ValueError(f"mesh scale must contain 3 numbers, got {value!r}")
  return np.asarray([float(part) for part in parts], dtype=np.float64)


def _load_obj_vertices(path: Path) -> np.ndarray:
  vertices: list[tuple[float, float, float]] = []
  for line in path.read_text(encoding="utf-8").splitlines():
    parts = line.strip().split()
    if len(parts) >= 4 and parts[0] == "v":
      vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
  if len(vertices) < 8:
    raise ValueError(f"OBJ box mesh must contain at least 8 vertices: {path}")
  return np.asarray(vertices, dtype=np.float64)


def _canonical_yaw(yaw: float) -> float:
  while yaw > math.pi / 2.0:
    yaw -= math.pi
  while yaw < -math.pi / 2.0:
    yaw += math.pi
  return yaw


def _fit_box(
  *,
  name: str,
  mesh_file: Path,
  vertices: np.ndarray,
  tolerance: float = 1.0e-5,
) -> OmniRetargetBox:
  center = vertices.mean(axis=0)
  xy = vertices[:, 0:2] - center[0:2]
  covariance = xy.T @ xy
  eigvals, eigvecs = np.linalg.eigh(covariance)
  axis = eigvecs[:, int(np.argmax(eigvals))]
  if axis[0] < 0.0:
    axis = -axis
  yaw = _canonical_yaw(float(math.atan2(axis[1], axis[0])))
  c = math.cos(yaw)
  s = math.sin(yaw)
  rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
  local_xy = xy @ rot
  local_z = vertices[:, 2] - center[2]
  half_xy = np.max(np.abs(local_xy), axis=0)
  half_z = float(np.max(np.abs(local_z)))
  if np.any(np.abs(local_xy) - half_xy > tolerance) or np.any(
    np.abs(local_z) - half_z > tolerance
  ):
    raise ValueError(f"mesh cannot be represented as one box: {mesh_file}")
  return OmniRetargetBox(
    name=name,
    mesh_file=mesh_file,
    pos=(float(center[0]), float(center[1]), float(center[2])),
    size=(float(half_xy[0]), float(half_xy[1]), half_z),
    yaw=yaw,
  )


def _iter_collision_meshes(
  urdf_path: Path,
) -> list[tuple[str, Path, np.ndarray]]:
  root = ET.parse(urdf_path).getroot()
  meshes: list[tuple[str, Path, np.ndarray]] = []
  seen: set[tuple[Path, tuple[float, float, float]]] = set()
  for idx, collision in enumerate(root.findall(".//collision")):
    mesh = collision.find(".//mesh")
    if mesh is None:
      continue
    filename = mesh.attrib.get("filename")
    if not filename:
      raise ValueError("collision mesh is missing filename")
    scale = _parse_scale(mesh.attrib.get("scale"))
    mesh_file = (urdf_path.parent / filename).resolve()
    if not mesh_file.exists():
      raise FileNotFoundError(mesh_file)
    key = (mesh_file, tuple(float(v) for v in scale))
    if key in seen:
      continue
    seen.add(key)
    name = collision.attrib.get("name") or f"box{idx}"
    meshes.append((name, mesh_file, scale))
  if not meshes:
    raise ValueError(f"terrain URDF does not contain collision meshes: {urdf_path}")
  return meshes


def load_omniretarget_terrain_boxes(
  urdf_path: str | Path,
) -> tuple[OmniRetargetBox, ...]:
  urdf_path = Path(urdf_path).resolve()
  boxes: list[OmniRetargetBox] = []
  for name, mesh_file, scale in _iter_collision_meshes(urdf_path):
    vertices = _load_obj_vertices(mesh_file) * scale
    boxes.append(_fit_box(name=name, mesh_file=mesh_file, vertices=vertices))
  return tuple(boxes)


def _yaw_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
  return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def _bounds_for_ground(
  boxes: tuple[OmniRetargetBox, ...],
  *,
  motion_file: str | Path | None,
  margin: float,
) -> tuple[np.ndarray, np.ndarray]:
  min_xy = []
  max_xy = []
  for box in boxes:
    radius = math.hypot(box.size[0], box.size[1])
    xy = np.asarray(box.pos[0:2], dtype=np.float64)
    min_xy.append(xy - radius)
    max_xy.append(xy + radius)
  if motion_file is not None:
    data = np.load(Path(motion_file))
    if "body_pos_w" in data:
      roots = np.asarray(data["body_pos_w"])[:, 0, 0:2]
      min_xy.append(np.min(roots, axis=0))
      max_xy.append(np.max(roots, axis=0))
  lower = np.min(np.asarray(min_xy), axis=0) - margin
  upper = np.max(np.asarray(max_xy), axis=0) + margin
  return lower, upper


def make_omniretarget_boxes_spec_fn(
  urdf_path: str | Path,
  *,
  motion_file: str | Path | None = None,
  ground_margin: float = 1.0,
  ground_depth: float = 0.1,
) -> Callable[[mujoco.MjSpec], None]:
  boxes = load_omniretarget_terrain_boxes(urdf_path)
  ground_min, ground_max = _bounds_for_ground(
    boxes,
    motion_file=motion_file,
    margin=ground_margin,
  )
  ground_center = 0.5 * (ground_min + ground_max)
  ground_size = 0.5 * (ground_max - ground_min)

  def spec_fn(spec: mujoco.MjSpec) -> None:
    body = spec.worldbody.add_body(name="omniretarget_terrain")
    ground = body.add_geom(
      name="omniretarget_ground",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=(float(ground_center[0]), float(ground_center[1]), -0.5 * ground_depth),
      size=(float(ground_size[0]), float(ground_size[1]), 0.5 * ground_depth),
      contype=1,
      conaffinity=1,
    )
    ground.mass = 0
    for box in boxes:
      geom = body.add_geom(
        name=f"omniretarget_{box.name}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=box.pos,
        size=box.size,
        quat=_yaw_quat_wxyz(box.yaw),
        contype=1,
        conaffinity=1,
      )
      geom.mass = 0

  return spec_fn
