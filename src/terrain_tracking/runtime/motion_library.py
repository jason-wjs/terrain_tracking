from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from terrain_tracking.runtime.pair_dataset import PairDatasetRecord


@dataclass(frozen=True)
class MotionLibrary:
  pair_ids: tuple[str, ...]
  pair_index_by_id: dict[str, int]
  frame_offsets: torch.Tensor
  frame_counts: torch.Tensor
  fps: torch.Tensor
  _joint_pos: torch.Tensor
  _joint_vel: torch.Tensor
  _body_pos_w: torch.Tensor
  _body_quat_w: torch.Tensor
  _body_lin_vel_w: torch.Tensor
  _body_ang_vel_w: torch.Tensor

  @classmethod
  def from_records(
    cls,
    records: Iterable[PairDatasetRecord],
    *,
    device: str | torch.device,
  ) -> "MotionLibrary":
    record_tuple = tuple(records)
    if not record_tuple:
      raise ValueError("MotionLibrary requires at least one record")

    loaded = [_load_motion(record.motion_file) for record in record_tuple]
    frame_counts_np = np.asarray([motion["joint_pos"].shape[0] for motion in loaded])
    frame_offsets_np = np.concatenate(
      [np.asarray([0], dtype=np.int64), np.cumsum(frame_counts_np[:-1], dtype=np.int64)]
    )
    torch_device = torch.device(device)

    return cls(
      pair_ids=tuple(record.pair_id for record in record_tuple),
      pair_index_by_id={
        record.pair_id: pair_index for pair_index, record in enumerate(record_tuple)
      },
      frame_offsets=torch.as_tensor(
        frame_offsets_np,
        dtype=torch.long,
        device=torch_device,
      ),
      frame_counts=torch.as_tensor(
        frame_counts_np,
        dtype=torch.long,
        device=torch_device,
      ),
      fps=torch.as_tensor(
        [record.motion.fps for record in record_tuple],
        dtype=torch.float32,
        device=torch_device,
      ),
      _joint_pos=_concat(loaded, "joint_pos", torch_device),
      _joint_vel=_concat(loaded, "joint_vel", torch_device),
      _body_pos_w=_concat(loaded, "body_pos_w", torch_device),
      _body_quat_w=_concat(loaded, "body_quat_w", torch_device),
      _body_lin_vel_w=_concat(loaded, "body_lin_vel_w", torch_device),
      _body_ang_vel_w=_concat(loaded, "body_ang_vel_w", torch_device),
    )

  def global_frame_index(
    self,
    *,
    pair_indices: torch.Tensor,
    local_frames: torch.Tensor,
  ) -> torch.Tensor:
    pair_indices = pair_indices.to(device=self.frame_offsets.device, dtype=torch.long)
    local_frames = local_frames.to(device=self.frame_offsets.device, dtype=torch.long)
    return self.frame_offsets[pair_indices] + local_frames

  def joint_pos(self, global_frame_indices: torch.Tensor) -> torch.Tensor:
    return self._joint_pos[global_frame_indices.to(self._joint_pos.device)]

  def joint_vel(self, global_frame_indices: torch.Tensor) -> torch.Tensor:
    return self._joint_vel[global_frame_indices.to(self._joint_vel.device)]

  def body_pos_w(self, global_frame_indices: torch.Tensor) -> torch.Tensor:
    return self._body_pos_w[global_frame_indices.to(self._body_pos_w.device)]

  def body_quat_w(self, global_frame_indices: torch.Tensor) -> torch.Tensor:
    return self._body_quat_w[global_frame_indices.to(self._body_quat_w.device)]

  def body_lin_vel_w(self, global_frame_indices: torch.Tensor) -> torch.Tensor:
    return self._body_lin_vel_w[global_frame_indices.to(self._body_lin_vel_w.device)]

  def body_ang_vel_w(self, global_frame_indices: torch.Tensor) -> torch.Tensor:
    return self._body_ang_vel_w[global_frame_indices.to(self._body_ang_vel_w.device)]


def _load_motion(path: Path) -> dict[str, np.ndarray]:
  required = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  )
  with np.load(path) as data:
    missing = [key for key in required if key not in data]
    if missing:
      raise ValueError(f"{path} missing motion arrays: {', '.join(missing)}")
    return {key: np.asarray(data[key], dtype=np.float32) for key in required}


def _concat(
  motions: list[dict[str, np.ndarray]],
  key: str,
  device: torch.device,
) -> torch.Tensor:
  return torch.as_tensor(
    np.concatenate([motion[key] for motion in motions], axis=0),
    dtype=torch.float32,
    device=device,
  )


__all__ = ["MotionLibrary"]
