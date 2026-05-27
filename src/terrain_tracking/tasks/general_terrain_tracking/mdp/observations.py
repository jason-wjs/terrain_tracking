from __future__ import annotations

from typing import Any

from torch import Tensor


def _body_index(command: Any, body_name: str) -> int:
  try:
    return command.cfg.body_names.index(body_name)
  except ValueError as exc:
    raise ValueError(
      f"Motion command body_names must include {body_name!r} for general teacher observations"
    ) from exc


def pelvis_pair_local_pos_w(env: Any, command_name: str) -> Tensor:
  command = env.command_manager.get_term(command_name)
  pelvis_idx = _body_index(command, "pelvis")
  return command.robot_body_pos_pair[:, pelvis_idx]


__all__ = ["pelvis_pair_local_pos_w"]
