from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def out_of_tile_bounds(
  env: ManagerBasedRlEnv,
  command_name: str,
  fail_margin: float = 1.0,
) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  root_pair_local_xy = (
    command.robot_anchor_pos_w[:, :2] - command.current_tile_origins[:, :2]
  )
  lower = command.current_occupied_bounds_xy[:, 0, :] - fail_margin
  upper = command.current_occupied_bounds_xy[:, 1, :] + fail_margin
  return torch.any((root_pair_local_xy < lower) | (root_pair_local_xy > upper), dim=1)


__all__ = ["out_of_tile_bounds"]
