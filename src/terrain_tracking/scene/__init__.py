"""Scene helpers for terrain_tracking."""

from .heightfield_spec import make_heightfield_spec_fn
from .pair_terrain_bank import PairTerrainBank
from .paired_mesh_spec import compute_env_origins_grid, make_paired_mesh_spec_fn
from .primitive_box_terrain import (
  PrimitiveBox,
  PrimitiveBoxDiagnostics,
  PrimitiveBoxTile,
  build_primitive_box_tile,
  make_primitive_box_tile_spec_fn,
)
from .terrain_tile import TerrainTile, bounds_from_boxes, union_bounds_xy

__all__ = [
  "compute_env_origins_grid",
  "PairTerrainBank",
  "PrimitiveBox",
  "PrimitiveBoxDiagnostics",
  "PrimitiveBoxTile",
  "build_primitive_box_tile",
  "make_heightfield_spec_fn",
  "make_paired_mesh_spec_fn",
  "make_primitive_box_tile_spec_fn",
  "TerrainTile",
  "bounds_from_boxes",
  "union_bounds_xy",
]
