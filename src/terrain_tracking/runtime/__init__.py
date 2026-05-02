"""Runtime helpers for terrain_tracking."""

from .pair_manifest import PairManifest
from .terrain_collision import TerrainCollisionManifest

__all__ = [
  "PairManifest",
  "TerrainCollisionManifest",
  "apply_pair_manifest_to_env_cfg",
]


def __getattr__(name: str):
  if name == "apply_pair_manifest_to_env_cfg":
    from .apply_pair import apply_pair_manifest_to_env_cfg

    return apply_pair_manifest_to_env_cfg
  raise AttributeError(name)
