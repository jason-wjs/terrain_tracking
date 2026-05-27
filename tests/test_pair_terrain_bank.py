from __future__ import annotations

import mujoco
import numpy as np

from terrain_tracking.scene.pair_terrain_bank import PairTerrainBank
from terrain_tracking.scene.primitive_box_terrain import PrimitiveBox
from terrain_tracking.scene.terrain_tile import TerrainTile


def _tile(pair_id: str, bounds: tuple[tuple[float, float], tuple[float, float]]):
  bounds_xy = np.asarray(bounds, dtype=np.float64)
  center = np.mean(bounds_xy, axis=0)
  half = 0.5 * (bounds_xy[1] - bounds_xy[0])
  return TerrainTile(
    pair_id=pair_id,
    boxes=(
      PrimitiveBox(
        name="terrain_base",
        pos=(float(center[0]), float(center[1]), -0.1),
        size=(float(half[0]), float(half[1]), 0.1),
      ),
    ),
    terrain_bounds_xy=bounds_xy,
    motion_root_bounds_xy=bounds_xy,
    occupied_bounds_xy=bounds_xy,
  )


def test_pair_terrain_bank_packs_tiles_and_adds_geoms_to_spec() -> None:
  tiles = (
    _tile("pair/a", ((0.0, 0.0), (1.0, 1.0))),
    _tile("pair/b", ((-1.0, -0.5), (1.0, 0.5))),
  )

  bank = PairTerrainBank.build(tiles, packing_margin=2.0)

  assert bank.tile_origins.shape == (2, 3)
  assert bank.pair_index_by_id["pair/a"] == 0
  assert bank.packing_cell_size[0] >= 6.0
  assert bank.packing_cell_size[1] >= 5.0

  spec = mujoco.MjSpec()
  bank.add_to_spec(spec)
  model = spec.compile()
  geom_names = {
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    for geom_id in range(model.ngeom)
  }

  assert "pair_a_terrain_base" in geom_names
  assert "pair_b_terrain_base" in geom_names
