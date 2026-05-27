from __future__ import annotations

from dataclasses import dataclass

import torch

from terrain_tracking.tasks.general_terrain_tracking.mdp.terminations import (
  out_of_tile_bounds,
)


@dataclass
class _FakeCommand:
  robot_anchor_pos_w: torch.Tensor
  current_tile_origins: torch.Tensor
  current_occupied_bounds_xy: torch.Tensor


class _FakeCommandManager:
  def __init__(self, command: _FakeCommand) -> None:
    self._command = command

  def get_term(self, name: str) -> _FakeCommand:
    assert name == "motion"
    return self._command


class _FakeEnv:
  def __init__(self, command: _FakeCommand) -> None:
    self.command_manager = _FakeCommandManager(command)


def test_out_of_tile_bounds_uses_pair_local_root_and_fail_margin() -> None:
  command = _FakeCommand(
    robot_anchor_pos_w=torch.tensor(
      [
        [1.5, 0.0, 0.8],
        [4.2, 0.0, 0.8],
      ]
    ),
    current_tile_origins=torch.tensor(
      [
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
      ]
    ),
    current_occupied_bounds_xy=torch.tensor(
      [
        [[0.0, -1.0], [1.0, 1.0]],
        [[0.0, -1.0], [1.0, 1.0]],
      ]
    ),
  )

  terminated = out_of_tile_bounds(_FakeEnv(command), command_name="motion", fail_margin=0.1)

  assert terminated.tolist() == [False, True]
