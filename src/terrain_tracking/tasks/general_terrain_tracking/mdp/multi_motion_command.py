from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_error_magnitude,
  quat_inv,
  quat_mul,
  yaw_quat,
)

from terrain_tracking.runtime.motion_library import MotionLibrary
from terrain_tracking.runtime.pair_dataset import PairDataset, ValidationMode
from terrain_tracking.runtime.pair_frame_sampler import (
  PairFrameSampler,
  PairFrameSamplerCfg,
)
from terrain_tracking.runtime.pair_selection import select_pair_records
from terrain_tracking.scene.pair_terrain_bank import PairTerrainBank

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _normalized_entropy(probabilities: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
  count = probabilities.shape[dim]
  entropy = -torch.sum(
    probabilities * torch.log(probabilities + 1.0e-12),
    dim=dim,
  )
  if count <= 1:
    return torch.ones_like(entropy)
  normalizer = torch.log(
    torch.tensor(
      float(count),
      device=probabilities.device,
      dtype=probabilities.dtype,
    )
  )
  return entropy / normalizer


class MultiMotionCommand(CommandTerm):
  cfg: MultiMotionCommandCfg

  def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    if not cfg.pair_dataset:
      raise ValueError("pair_dataset is required for MultiMotionCommand")
    self.robot = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    dataset = PairDataset.load(cfg.pair_dataset, validate=cfg.dataset_validate)
    self.records = select_pair_records(
      dataset.records,
      pair_filter=cfg.pair_filter,
      max_pairs=cfg.max_pairs,
    )
    self.motion = MotionLibrary.from_records(self.records, device=self.device)
    self.bank = PairTerrainBank.from_records(
      self.records,
      packing_margin=cfg.packing_margin,
    )
    pair_weights = torch.tensor(
      [record.weight for record in self.records],
      dtype=torch.float32,
      device=self.device,
    )
    self.sampler = PairFrameSampler(
      frame_counts=self.motion.frame_counts,
      cfg=cfg.sampler,
      device=self.device,
      pair_weights=pair_weights,
    )
    self._bank_tile_origins = torch.as_tensor(
      self.bank.tile_origins,
      dtype=torch.float32,
      device=self.device,
    )
    self._bank_occupied_bounds_xy = torch.as_tensor(
      self.bank.occupied_bounds_xy,
      dtype=torch.float32,
      device=self.device,
    )
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.env_pair_indices = torch.zeros(
      self.num_envs,
      dtype=torch.long,
      device=self.device,
    )
    self.current_tile_origins = torch.zeros(self.num_envs, 3, device=self.device)
    self.current_occupied_bounds_xy = torch.zeros(self.num_envs, 2, 2, device=self.device)
    for metric_name in (
      "sampling_entropy",
      "sampling_top1_prob",
      "sampling_pair_entropy",
      "sampling_pair_top1_prob",
      "sampling_bin_entropy_mean",
      "sampling_bin_entropy_min",
      "sampling_bin_top1_prob_mean",
      "sampling_bin_top1_prob_max",
      "active_pair_entropy",
      "active_pair_top1_frac",
    ):
      self.metrics[metric_name] = torch.zeros(self.num_envs, device=self.device)
    for metric_name in (
      "error_anchor_pos",
      "error_anchor_rot",
      "error_anchor_lin_vel",
      "error_anchor_ang_vel",
      "error_body_pos",
      "error_body_rot",
      "error_body_lin_vel",
      "error_body_ang_vel",
      "error_joint_pos",
      "error_joint_vel",
    ):
      self.metrics[metric_name] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def _global_frame_indices(self) -> torch.Tensor:
    return self.motion.global_frame_index(
      pair_indices=self.env_pair_indices,
      local_frames=self.time_steps,
    )

  @property
  def env_pair_ids(self) -> tuple[str, ...]:
    return tuple(self.motion.pair_ids[int(index)] for index in self.env_pair_indices.cpu())

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos(self._global_frame_indices)

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel(self._global_frame_indices)

  @property
  def body_pos_w(self) -> torch.Tensor:
    body_pos = self.motion.body_pos_w(self._global_frame_indices)[:, self.body_indexes]
    return body_pos + self.current_tile_origins[:, None, :]

  @property
  def body_pos_pair(self) -> torch.Tensor:
    return self.body_pos_w - self.current_tile_origins[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w(self._global_frame_indices)[:, self.body_indexes]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w(self._global_frame_indices)[:, self.body_indexes]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w(self._global_frame_indices)[:, self.body_indexes]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return self.body_pos_w[:, self.motion_anchor_body_index]

  @property
  def anchor_pos_pair(self) -> torch.Tensor:
    return self.anchor_pos_w - self.current_tile_origins

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.body_quat_w[:, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.body_lin_vel_w[:, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.body_ang_vel_w[:, self.motion_anchor_body_index]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_pos_pair(self) -> torch.Tensor:
    return self.robot_body_pos_w - self.current_tile_origins[:, None, :]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_pos_pair(self) -> torch.Tensor:
    return self.robot_anchor_pos_w - self.current_tile_origins

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self) -> None:
    pair_probs = self.sampler.pair_probabilities
    bin_probs = self.sampler.bin_weights
    joint_probs = self.sampler.joint_bin_probabilities.reshape(-1)

    pair_entropy = _normalized_entropy(pair_probs)
    pair_top1 = torch.max(pair_probs)
    bin_entropy = _normalized_entropy(bin_probs, dim=1)
    bin_top1 = torch.max(bin_probs, dim=1).values
    joint_entropy = _normalized_entropy(joint_probs)
    joint_top1 = torch.max(joint_probs)

    active_counts = torch.bincount(
      self.env_pair_indices,
      minlength=int(self.motion.frame_counts.numel()),
    ).to(dtype=torch.float32)
    active_probs = active_counts / torch.clamp(active_counts.sum(), min=1.0)
    active_entropy = _normalized_entropy(active_probs)
    active_top1 = torch.max(active_probs)

    self.metrics["sampling_entropy"][:] = joint_entropy
    self.metrics["sampling_top1_prob"][:] = joint_top1
    self.metrics["sampling_pair_entropy"][:] = pair_entropy
    self.metrics["sampling_pair_top1_prob"][:] = pair_top1
    self.metrics["sampling_bin_entropy_mean"][:] = torch.mean(bin_entropy)
    self.metrics["sampling_bin_entropy_min"][:] = torch.min(bin_entropy)
    self.metrics["sampling_bin_top1_prob_mean"][:] = torch.mean(bin_top1)
    self.metrics["sampling_bin_top1_prob_max"][:] = torch.max(bin_top1)
    self.metrics["active_pair_entropy"][:] = active_entropy
    self.metrics["active_pair_top1_frac"][:] = active_top1
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w,
      dim=-1,
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w,
      self.robot_anchor_quat_w,
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w,
      dim=-1,
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w,
      dim=-1,
    )
    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w,
      dim=-1,
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w,
      self.robot_body_quat_w,
    ).mean(dim=-1)
    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w,
      dim=-1,
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w,
      dim=-1,
    ).mean(dim=-1)
    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos,
      dim=-1,
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel,
      dim=-1,
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.cfg.sampling_mode == "start":
      sample = self.sampler.sample(env_ids)
      pair_indices = sample.pair_indices
      local_frames = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
    elif self.cfg.sampling_mode == "uniform":
      sample = self.sampler.sample(env_ids)
      pair_indices = sample.pair_indices
      local_frames = sample.local_frames
    else:
      assert self.cfg.sampling_mode == "adaptive"
      if self.cfg.sampling_mode == "adaptive" and hasattr(
        self._env,
        "termination_manager",
      ):
        self.sampler.update_failures(
          self.env_pair_indices[env_ids],
          self.time_steps[env_ids],
          self._env.termination_manager.terminated[env_ids],
        )
      sample = self.sampler.sample(env_ids)
      pair_indices = sample.pair_indices
      local_frames = sample.local_frames

    self.env_pair_indices[env_ids] = pair_indices
    self.time_steps[env_ids] = local_frames
    self.current_tile_origins[env_ids] = self._bank_tile_origins[pair_indices]
    self.current_occupied_bounds_xy[env_ids] = self._bank_occupied_bounds_xy[
      pair_indices
    ]

    self._write_reference_state_to_sim(
      env_ids,
      self.body_pos_w[env_ids, 0],
      self.body_quat_w[env_ids, 0],
      self.body_lin_vel_w[env_ids, 0],
      self.body_ang_vel_w[env_ids, 0],
      self.joint_pos[env_ids],
      self.joint_vel[env_ids],
    )
    self.update_relative_body_poses()

  def _update_command(self) -> None:
    self.time_steps += 1
    expired = self.time_steps >= self.motion.frame_counts[self.env_pair_indices]
    env_ids = torch.where(expired)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)
    self.update_relative_body_poses()

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def update_relative_body_poses(self) -> None:
    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1,
      len(self.cfg.body_names),
      1,
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1,
      len(self.cfg.body_names),
      1,
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1,
      len(self.cfg.body_names),
      1,
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1,
      len(self.cfg.body_names),
      1,
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w,
      self.body_pos_w - anchor_pos_w_repeat,
    )


@dataclass(kw_only=True)
class MultiMotionCommandCfg(CommandTermCfg):
  pair_dataset: str
  entity_name: str
  anchor_body_name: str
  body_names: tuple[str, ...]
  dataset_validate: ValidationMode = "fast"
  terrain_adapter: Literal["parc_primitive_boxes"] = "parc_primitive_boxes"
  packing_margin: float = 2.0
  fail_margin: float = 1.0
  max_pairs: int | None = None
  pair_filter: tuple[str, ...] = ()
  sampler: PairFrameSamplerCfg = field(default_factory=PairFrameSamplerCfg)
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"

  def build(self, env: ManagerBasedRlEnv) -> MultiMotionCommand:
    return MultiMotionCommand(self, env)


__all__ = [
  "MultiMotionCommand",
  "MultiMotionCommandCfg",
]
