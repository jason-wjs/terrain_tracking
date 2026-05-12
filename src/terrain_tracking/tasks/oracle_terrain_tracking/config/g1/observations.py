from __future__ import annotations

from typing import Any, cast

from torch import Tensor

from mjlab.tasks.tracking.mdp import MotionCommand


def _motion_command(env: Any, command_name: str) -> MotionCommand:
  return cast(MotionCommand, env.command_manager.get_term(command_name))


def global_anchor_pos_error_w(env: Any, command_name: str) -> Tensor:
  command = _motion_command(env, command_name)
  return command.anchor_pos_w - command.robot_anchor_pos_w


def global_anchor_lin_vel_error_w(env: Any, command_name: str) -> Tensor:
  command = _motion_command(env, command_name)
  return command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w


def reference_anchor_lin_vel_w(env: Any, command_name: str) -> Tensor:
  command = _motion_command(env, command_name)
  return command.anchor_lin_vel_w
