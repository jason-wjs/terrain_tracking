from __future__ import annotations

from typing import Any, cast

from mjlab.tasks.tracking.mdp import MotionCommand
from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms
from torch import Tensor


def _motion_command(env: Any, command_name: str) -> MotionCommand:
  return cast(MotionCommand, env.command_manager.get_term(command_name))


def _body_index(command: MotionCommand, body_name: str) -> int:
  try:
    return command.cfg.body_names.index(body_name)
  except ValueError as exc:
    raise ValueError(
      f"Motion command body_names must include {body_name!r} for PHP teacher observations"
    ) from exc


def _pelvis_index(command: MotionCommand) -> int:
  return _body_index(command, "pelvis")


def reference_pelvis_pos_error_b(env: Any, command_name: str) -> Tensor:
  command = _motion_command(env, command_name)
  pelvis_idx = _pelvis_index(command)
  pos, _ = subtract_frame_transforms(
    command.robot_body_pos_w[:, pelvis_idx],
    command.robot_body_quat_w[:, pelvis_idx],
    command.body_pos_w[:, pelvis_idx],
    command.body_quat_w[:, pelvis_idx],
  )
  return pos.view(env.num_envs, -1)


def reference_pelvis_ori_error_b(env: Any, command_name: str) -> Tensor:
  command = _motion_command(env, command_name)
  pelvis_idx = _pelvis_index(command)
  _, ori = subtract_frame_transforms(
    command.robot_body_pos_w[:, pelvis_idx],
    command.robot_body_quat_w[:, pelvis_idx],
    command.body_pos_w[:, pelvis_idx],
    command.body_quat_w[:, pelvis_idx],
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def pelvis_lin_vel(env: Any, command_name: str) -> Tensor:
  command = _motion_command(env, command_name)
  pelvis_idx = _pelvis_index(command)
  return command.robot_body_lin_vel_w[:, pelvis_idx]


def pelvis_ang_vel(env: Any, command_name: str) -> Tensor:
  command = _motion_command(env, command_name)
  pelvis_idx = _pelvis_index(command)
  return command.robot_body_ang_vel_w[:, pelvis_idx]


def pelvis_global_pos_w(env: Any, command_name: str) -> Tensor:
  command = _motion_command(env, command_name)
  pelvis_idx = _pelvis_index(command)
  return command.robot_body_pos_w[:, pelvis_idx]


def pelvis_global_lin_vel_w(env: Any, command_name: str) -> Tensor:
  command = _motion_command(env, command_name)
  pelvis_idx = _pelvis_index(command)
  return command.robot_body_lin_vel_w[:, pelvis_idx]


def global_anchor_pos_error_w(env: Any, command_name: str) -> Tensor:
  command = _motion_command(env, command_name)
  return command.anchor_pos_w - command.robot_anchor_pos_w


def global_anchor_lin_vel_error_w(env: Any, command_name: str) -> Tensor:
  command = _motion_command(env, command_name)
  return command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w


def reference_anchor_lin_vel_w(env: Any, command_name: str) -> Tensor:
  command = _motion_command(env, command_name)
  return command.anchor_lin_vel_w
