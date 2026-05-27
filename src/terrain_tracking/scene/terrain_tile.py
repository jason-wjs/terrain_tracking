from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from terrain_tracking.scene.primitive_box_terrain import PrimitiveBox


@dataclass(frozen=True)
class TerrainTile:
  pair_id: str
  boxes: tuple[PrimitiveBox, ...]
  terrain_bounds_xy: NDArray[np.float64]
  motion_root_bounds_xy: NDArray[np.float64]
  occupied_bounds_xy: NDArray[np.float64]

  @property
  def bounds_xy(self) -> NDArray[np.float64]:
    return self.occupied_bounds_xy


def bounds_from_boxes(boxes: tuple[PrimitiveBox, ...]) -> NDArray[np.float64]:
  if not boxes:
    raise ValueError("terrain tile must contain at least one box")
  mins = []
  maxs = []
  for box in boxes:
    pos = np.asarray(box.pos[:2], dtype=np.float64)
    size = np.asarray(box.size[:2], dtype=np.float64)
    mins.append(pos - size)
    maxs.append(pos + size)
  return np.stack(
    [np.min(np.stack(mins, axis=0), axis=0), np.max(np.stack(maxs, axis=0), axis=0)],
    axis=0,
  )


def union_bounds_xy(*bounds: NDArray[np.float64]) -> NDArray[np.float64]:
  if not bounds:
    raise ValueError("at least one bounds array is required")
  stacked = np.stack(bounds, axis=0)
  return np.stack(
    [np.min(stacked[:, 0, :], axis=0), np.max(stacked[:, 1, :], axis=0)],
    axis=0,
  )


__all__ = [
  "TerrainTile",
  "bounds_from_boxes",
  "union_bounds_xy",
]
