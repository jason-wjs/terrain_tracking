from __future__ import annotations

from typing import Protocol

from terrain_tracking.runtime.pair_dataset import PairDatasetRecord
from terrain_tracking.scene.terrain_tile import TerrainTile


class TerrainAdapter(Protocol):
  def build_tile(self, record: PairDatasetRecord) -> TerrainTile: ...


__all__ = ["TerrainAdapter"]
