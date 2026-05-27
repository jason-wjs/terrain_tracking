from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from terrain_tracking.runtime.pair_dataset import PairDataset
from terrain_tracking.runtime.pair_selection import select_pair_records
from terrain_tracking.scene.pair_terrain_bank import PairTerrainBank
from terrain_tracking.tasks.general_terrain_tracking.mdp import MultiMotionCommandCfg


def apply_pair_dataset_to_env_cfg(cfg: ManagerBasedRlEnvCfg) -> PairTerrainBank:
  motion_cmd = cfg.commands["motion"]
  if not isinstance(motion_cmd, MultiMotionCommandCfg):
    raise TypeError("Expected cfg.commands['motion'] to be a MultiMotionCommandCfg")
  if not motion_cmd.pair_dataset:
    raise ValueError("pair_dataset is required for the general terrain teacher task")

  dataset = PairDataset.load(motion_cmd.pair_dataset, validate=motion_cmd.dataset_validate)
  records = select_pair_records(
    dataset.records,
    pair_filter=motion_cmd.pair_filter,
    max_pairs=motion_cmd.max_pairs,
  )
  bank = PairTerrainBank.from_records(records, packing_margin=motion_cmd.packing_margin)
  existing_spec_fn = cfg.scene.spec_fn

  def spec_fn(spec) -> None:
    if existing_spec_fn is not None:
      existing_spec_fn(spec)
    bank.add_to_spec(spec)

  cfg.scene.spec_fn = spec_fn
  cfg.scene.env_spacing = 0.0
  total_boxes = sum(len(tile.boxes) for tile in bank.tiles)
  if cfg.sim.nconmax is None or cfg.sim.nconmax < max(256, total_boxes * 4):
    cfg.sim.nconmax = max(256, total_boxes * 4)
  if cfg.sim.njmax is None or cfg.sim.njmax < max(512, total_boxes * 8):
    cfg.sim.njmax = max(512, total_boxes * 8)
  return bank


__all__ = ["apply_pair_dataset_to_env_cfg"]
