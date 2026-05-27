from __future__ import annotations

import numpy as np

from terrain_tracking.runtime.pair_dataset import PairDatasetRecord
from terrain_tracking.runtime.terrain_collision import TerrainCollisionManifest
from terrain_tracking.scene.primitive_box_terrain import build_primitive_box_tile
from terrain_tracking.scene.terrain_tile import (
  TerrainTile,
  bounds_from_boxes,
  union_bounds_xy,
)


class ParcPrimitiveBoxAdapter:
  adapter_name = "parc_primitive_boxes"

  def build_tile(self, record: PairDatasetRecord) -> TerrainTile:
    if record.terrain.adapter != self.adapter_name:
      raise ValueError(
        f"record {record.pair_id} uses adapter {record.terrain.adapter!r}, "
        f"expected {self.adapter_name!r}"
      )
    manifest = TerrainCollisionManifest.load(record.terrain_collision_file)
    primitive_tile = build_primitive_box_tile(manifest)
    if primitive_tile.diagnostics.box_count != record.terrain.box_count:
      raise ValueError(
        f"record {record.pair_id} terrain box_count is stale: "
        f"manifest={record.terrain.box_count}, actual={primitive_tile.diagnostics.box_count}"
      )

    terrain_bounds_xy = bounds_from_boxes(primitive_tile.boxes)
    motion_root_bounds_xy = np.asarray(record.motion.root_bounds_xy, dtype=np.float64)
    return TerrainTile(
      pair_id=record.pair_id,
      boxes=primitive_tile.boxes,
      terrain_bounds_xy=terrain_bounds_xy,
      motion_root_bounds_xy=motion_root_bounds_xy,
      occupied_bounds_xy=union_bounds_xy(terrain_bounds_xy, motion_root_bounds_xy),
    )


__all__ = ["ParcPrimitiveBoxAdapter"]
