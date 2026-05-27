from __future__ import annotations

import math
import re
from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

from terrain_tracking.runtime.pair_dataset import PairDatasetRecord
from terrain_tracking.scene.terrain_adapters.parc import ParcPrimitiveBoxAdapter
from terrain_tracking.scene.terrain_tile import TerrainTile


@dataclass(frozen=True)
class PairTerrainBank:
  tiles: tuple[TerrainTile, ...]
  tile_origins: NDArray[np.float64]
  occupied_bounds_xy: NDArray[np.float64]
  packing_bounds_xy: NDArray[np.float64]
  packing_cell_size: NDArray[np.float64]
  pair_index_by_id: dict[str, int]
  packing_margin: float

  @classmethod
  def from_records(
    cls,
    records: tuple[PairDatasetRecord, ...] | list[PairDatasetRecord],
    *,
    packing_margin: float = 2.0,
  ) -> "PairTerrainBank":
    adapter = ParcPrimitiveBoxAdapter()
    return cls.build(
      [adapter.build_tile(record) for record in records],
      packing_margin=packing_margin,
    )

  @classmethod
  def build(
    cls,
    tiles: tuple[TerrainTile, ...] | list[TerrainTile],
    *,
    packing_margin: float = 2.0,
  ) -> "PairTerrainBank":
    tile_tuple = tuple(tiles)
    if not tile_tuple:
      raise ValueError("PairTerrainBank requires at least one tile")
    if packing_margin < 0.0:
      raise ValueError("packing_margin must be non-negative")

    occupied_bounds_xy = np.stack(
      [tile.occupied_bounds_xy for tile in tile_tuple],
      axis=0,
    ).astype(np.float64)
    padding = np.asarray([packing_margin, packing_margin], dtype=np.float64)
    packing_bounds_xy = occupied_bounds_xy.copy()
    packing_bounds_xy[:, 0, :] -= padding
    packing_bounds_xy[:, 1, :] += padding

    packing_sizes = packing_bounds_xy[:, 1, :] - packing_bounds_xy[:, 0, :]
    packing_cell_size = np.max(packing_sizes, axis=0)
    cols = max(1, math.ceil(math.sqrt(len(tile_tuple))))
    tile_origins = np.zeros((len(tile_tuple), 3), dtype=np.float64)
    for tile_index, packing_bounds in enumerate(packing_bounds_xy):
      row = tile_index // cols
      col = tile_index % cols
      cell_min = np.asarray(
        [col * packing_cell_size[0], row * packing_cell_size[1]],
        dtype=np.float64,
      )
      tile_origins[tile_index, :2] = cell_min - packing_bounds[0]

    return cls(
      tiles=tile_tuple,
      tile_origins=tile_origins,
      occupied_bounds_xy=occupied_bounds_xy,
      packing_bounds_xy=packing_bounds_xy,
      packing_cell_size=packing_cell_size,
      pair_index_by_id={tile.pair_id: idx for idx, tile in enumerate(tile_tuple)},
      packing_margin=float(packing_margin),
    )

  def add_to_spec(self, spec: mujoco.MjSpec) -> None:
    for tile_index, tile in enumerate(self.tiles):
      origin = self.tile_origins[tile_index]
      for box in tile.boxes:
        geom = spec.worldbody.add_geom(
          name=f"{_sanitize_name(tile.pair_id)}_{_sanitize_name(box.name)}",
          type=mujoco.mjtGeom.mjGEOM_BOX,
          pos=(
            box.pos[0] + float(origin[0]),
            box.pos[1] + float(origin[1]),
            box.pos[2] + float(origin[2]),
          ),
          size=box.size,
          contype=1,
          conaffinity=1,
        )
        geom.mass = 0

  def tile_origin_for_pair_indices(
    self,
    pair_indices: NDArray[np.integer],
  ) -> NDArray[np.float64]:
    return self.tile_origins[pair_indices]


def _sanitize_name(value: str) -> str:
  sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")
  return sanitized or "unnamed"


__all__ = ["PairTerrainBank"]
