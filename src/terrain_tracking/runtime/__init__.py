"""Runtime helpers for terrain_tracking."""

from typing import TYPE_CHECKING

from .motion_library import MotionLibrary
from .pair_dataset import PairDataset, PairDatasetRecord
from .pair_frame_sampler import PairFrameSampler, PairFrameSamplerCfg
from .pair_manifest import PairManifest
from .pair_selection import select_pair_records
from .terrain_collision import TerrainCollisionManifest

if TYPE_CHECKING:
  from .apply_pair import apply_pair_manifest_to_env_cfg
  from .apply_pair_dataset import apply_pair_dataset_to_env_cfg

__all__ = [
  "PairDataset",
  "PairDatasetRecord",
  "MotionLibrary",
  "PairFrameSampler",
  "PairFrameSamplerCfg",
  "PairManifest",
  "select_pair_records",
  "TerrainCollisionManifest",
  "apply_pair_manifest_to_env_cfg",
  "apply_pair_dataset_to_env_cfg",
]


def __getattr__(name: str):
  if name == "apply_pair_dataset_to_env_cfg":
    from .apply_pair_dataset import apply_pair_dataset_to_env_cfg

    return apply_pair_dataset_to_env_cfg
  if name == "apply_pair_manifest_to_env_cfg":
    from .apply_pair import apply_pair_manifest_to_env_cfg

    return apply_pair_manifest_to_env_cfg
  raise AttributeError(name)
