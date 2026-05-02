from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from terrain_tracking.runtime.terrain_collision import TerrainCollisionManifest


@dataclass(frozen=True)
class PrimitiveBox:
  name: str
  pos: tuple[float, float, float]
  size: tuple[float, float, float]


@dataclass(frozen=True)
class PrimitiveBoxDiagnostics:
  hf_shape: tuple[int, int]
  unique_height_count: int
  nonzero_cell_count: int
  box_count: int


@dataclass(frozen=True)
class PrimitiveBoxTile:
  boxes: tuple[PrimitiveBox, ...]
  diagnostics: PrimitiveBoxDiagnostics
  footprint: tuple[float, float]


def _load_scaled_hf(manifest: TerrainCollisionManifest) -> np.ndarray:
  hf = np.load(manifest.hf_file).astype(np.float32)
  if hf.ndim != 2:
    raise ValueError(f"heightfield must be 2D, got shape {hf.shape}")
  return hf * np.float32(manifest.height_scale)


def _greedy_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
  used = np.zeros(mask.shape, dtype=bool)
  rects: list[tuple[int, int, int, int]] = []
  nx, ny = mask.shape
  for i0 in range(nx):
    for j0 in range(ny):
      if used[i0, j0] or not mask[i0, j0]:
        continue
      j1 = j0
      while j1 + 1 < ny and mask[i0, j1 + 1] and not used[i0, j1 + 1]:
        j1 += 1
      i1 = i0
      while i1 + 1 < nx:
        row = mask[i1 + 1, j0 : j1 + 1]
        row_used = used[i1 + 1, j0 : j1 + 1]
        if not bool(np.all(row & ~row_used)):
          break
        i1 += 1
      used[i0 : i1 + 1, j0 : j1 + 1] = True
      rects.append((i0, i1, j0, j1))
  return rects


def _rect_to_box(
  *,
  name: str,
  i0: int,
  i1: int,
  j0: int,
  j1: int,
  height: float,
  manifest: TerrainCollisionManifest,
) -> PrimitiveBox:
  cell = float(manifest.dx * manifest.xy_scale)
  min_x = float(manifest.min_point[0] * manifest.xy_scale)
  min_y = float(manifest.min_point[1] * manifest.xy_scale)
  sx = 0.5 * float(i1 - i0 + 1) * cell
  sy = 0.5 * float(j1 - j0 + 1) * cell
  sz = 0.5 * float(height)
  cx = min_x + float(i0 + i1) * 0.5 * cell
  cy = min_y + float(j0 + j1) * 0.5 * cell
  cz = sz
  return PrimitiveBox(name=name, pos=(cx, cy, cz), size=(sx, sy, sz))


def _shift_box(box: PrimitiveBox, offset: tuple[float, float, float]) -> PrimitiveBox:
  return PrimitiveBox(
    name=box.name,
    pos=(
      box.pos[0] + offset[0],
      box.pos[1] + offset[1],
      box.pos[2] + offset[2],
    ),
    size=box.size,
  )


def build_primitive_box_tile(
  manifest: TerrainCollisionManifest,
  *,
  height_epsilon: float = 1.0e-5,
  max_boxes: int = 128,
) -> PrimitiveBoxTile:
  hf = _load_scaled_hf(manifest)
  nx, ny = hf.shape
  footprint = (
    float(nx * manifest.dx * manifest.xy_scale),
    float(ny * manifest.dx * manifest.xy_scale),
  )
  cell = float(manifest.dx * manifest.xy_scale)
  min_x = float(manifest.min_point[0] * manifest.xy_scale)
  min_y = float(manifest.min_point[1] * manifest.xy_scale)
  center_x = min_x + 0.5 * float(nx - 1) * cell
  center_y = min_y + 0.5 * float(ny - 1) * cell
  base_depth = max(abs(float(manifest.base_z)), 1.0e-6)
  base = PrimitiveBox(
    name="terrain_base",
    pos=(center_x, center_y, -0.5 * base_depth),
    size=(0.5 * footprint[0], 0.5 * footprint[1], 0.5 * base_depth),
  )

  heights = np.unique(hf[hf > height_epsilon])
  boxes = [base]
  for height_idx, height in enumerate(heights):
    mask = np.isclose(hf, height, atol=height_epsilon)
    for rect_idx, (i0, i1, j0, j1) in enumerate(_greedy_rectangles(mask)):
      boxes.append(
        _rect_to_box(
          name=f"terrain_h{height_idx}_rect{rect_idx}",
          i0=i0,
          i1=i1,
          j0=j0,
          j1=j1,
          height=float(height),
          manifest=manifest,
        )
      )

  if len(boxes) > max_boxes:
    raise ValueError(
      "primitive box terrain is too complex: "
      f"box_count={len(boxes)}, max_boxes={max_boxes}, hf_shape={hf.shape}"
    )

  diagnostics = PrimitiveBoxDiagnostics(
    hf_shape=(int(nx), int(ny)),
    unique_height_count=int(len(np.unique(hf))),
    nonzero_cell_count=int(np.count_nonzero(hf > height_epsilon)),
    box_count=len(boxes),
  )
  return PrimitiveBoxTile(
    boxes=tuple(boxes),
    diagnostics=diagnostics,
    footprint=footprint,
  )


def make_primitive_box_tile_spec_fn(
  manifest: TerrainCollisionManifest,
  *,
  max_boxes: int = 128,
  local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Callable[[mujoco.MjSpec, mujoco.MjsBody], PrimitiveBoxTile]:
  tile = build_primitive_box_tile(manifest, max_boxes=max_boxes)
  shifted_boxes = tuple(_shift_box(box, local_offset) for box in tile.boxes)

  def spec_fn(_spec: mujoco.MjSpec, body: mujoco.MjsBody) -> PrimitiveBoxTile:
    for box in shifted_boxes:
      geom = body.add_geom(
        name=box.name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=box.pos,
        size=box.size,
        contype=1,
        conaffinity=1,
      )
      geom.mass = 0
    return tile

  return spec_fn
