from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML


@dataclass(frozen=True)
class OmniRetargetClip:
  qpos: np.ndarray
  fps: int
  times: np.ndarray

  @property
  def root_quat_wxyz(self) -> np.ndarray:
    return self.qpos[:, 0:4]

  @property
  def root_pos_w(self) -> np.ndarray:
    return self.qpos[:, 4:7]

  @property
  def joint_pos(self) -> np.ndarray:
    return self.qpos[:, 7:36]


def _as_scalar_int(value: np.ndarray, *, key: str) -> int:
  array = np.asarray(value)
  if array.shape != ():
    raise ValueError(f"{key} must be a scalar")
  return int(array.item())


def _validate_qpos(qpos: np.ndarray) -> np.ndarray:
  qpos = np.asarray(qpos, dtype=np.float64)
  if qpos.ndim != 2 or qpos.shape[1] != 36:
    raise ValueError(f"qpos must have shape [T, 36], got {qpos.shape}")
  if qpos.shape[0] < 2:
    raise ValueError("qpos must contain at least two frames")
  return qpos


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
  norm = np.linalg.norm(quat, axis=-1, keepdims=True)
  if np.any(norm == 0.0):
    raise ValueError("root quaternion must be non-zero")
  return quat / norm


def load_omniretarget_clip(path: str | Path) -> OmniRetargetClip:
  data = np.load(Path(path))
  if "qpos" not in data:
    raise ValueError("OmniRetarget clip is missing qpos")
  if "fps" not in data:
    raise ValueError("OmniRetarget clip is missing fps")

  qpos = _validate_qpos(data["qpos"])
  qpos = qpos.copy()
  qpos[:, 0:4] = _normalize_quat_wxyz(qpos[:, 0:4])
  fps = _as_scalar_int(data["fps"], key="fps")
  if fps <= 0:
    raise ValueError(f"fps must be positive, got {fps}")
  times = np.arange(qpos.shape[0], dtype=np.float64) / float(fps)
  return OmniRetargetClip(qpos=qpos, fps=fps, times=times)


def _interp_quat_nlerp(
  source_times: np.ndarray,
  quats: np.ndarray,
  target_times: np.ndarray,
) -> np.ndarray:
  quats = _normalize_quat_wxyz(quats)
  output = np.empty((target_times.shape[0], 4), dtype=np.float64)
  for out_idx, target_time in enumerate(target_times):
    right = int(np.searchsorted(source_times, target_time, side="right"))
    if right <= 0:
      output[out_idx] = quats[0]
      continue
    if right >= source_times.shape[0]:
      output[out_idx] = quats[-1]
      continue
    left = right - 1
    span = source_times[right] - source_times[left]
    alpha = 0.0 if span == 0.0 else (target_time - source_times[left]) / span
    q0 = quats[left]
    q1 = quats[right]
    if float(np.dot(q0, q1)) < 0.0:
      q1 = -q1
    output[out_idx] = (1.0 - alpha) * q0 + alpha * q1
  return _normalize_quat_wxyz(output)


def _interp_columns(
  source_times: np.ndarray,
  values: np.ndarray,
  target_times: np.ndarray,
) -> np.ndarray:
  output = np.empty((target_times.shape[0], values.shape[1]), dtype=np.float64)
  for col in range(values.shape[1]):
    output[:, col] = np.interp(target_times, source_times, values[:, col])
  return output


def resample_clip(
  clip: OmniRetargetClip,
  *,
  output_fps: int = 50,
) -> OmniRetargetClip:
  if output_fps <= 0:
    raise ValueError(f"output_fps must be positive, got {output_fps}")
  duration = float(clip.times[-1])
  output_frame_count = int(round(duration * output_fps)) + 1
  target_times = np.linspace(0.0, duration, output_frame_count, dtype=np.float64)

  root_quat = _interp_quat_nlerp(
    clip.times,
    clip.root_quat_wxyz,
    target_times,
  )
  root_pos = _interp_columns(clip.times, clip.root_pos_w, target_times)
  joint_pos = _interp_columns(clip.times, clip.joint_pos, target_times)
  qpos = np.concatenate([root_quat, root_pos, joint_pos], axis=1)
  return OmniRetargetClip(qpos=qpos, fps=output_fps, times=target_times)


def _to_mujoco_qpos(clip_qpos: np.ndarray) -> np.ndarray:
  mujoco_qpos = np.empty_like(clip_qpos, dtype=np.float64)
  mujoco_qpos[:, 0:3] = clip_qpos[:, 4:7]
  mujoco_qpos[:, 3:7] = clip_qpos[:, 0:4]
  mujoco_qpos[:, 7:36] = clip_qpos[:, 7:36]
  return mujoco_qpos


def _differentiate_qpos(
  model: mujoco.MjModel,
  qpos: np.ndarray,
  dt: float,
) -> np.ndarray:
  qvel = np.zeros((qpos.shape[0], model.nv), dtype=np.float64)
  for frame in range(qpos.shape[0]):
    if frame == 0:
      q0, q1, denom = qpos[0], qpos[1], dt
    elif frame == qpos.shape[0] - 1:
      q0, q1, denom = qpos[-2], qpos[-1], dt
    else:
      q0, q1, denom = qpos[frame - 1], qpos[frame + 1], 2.0 * dt
    mujoco.mj_differentiatePos(model, qvel[frame], denom, q0, q1)
  return qvel


def _quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
  result = quat.copy()
  result[..., 1:] *= -1.0
  return result


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  aw, ax, ay, az = np.moveaxis(a, -1, 0)
  bw, bx, by, bz = np.moveaxis(b, -1, 0)
  return np.stack(
    [
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ],
    axis=-1,
  )


def _angular_velocity_from_quats(quats: np.ndarray, dt: float) -> np.ndarray:
  velocities = np.zeros((quats.shape[0], quats.shape[1], 3), dtype=np.float32)
  for frame in range(quats.shape[0]):
    if frame == 0:
      q0, q1, denom = quats[0], quats[1], dt
    elif frame == quats.shape[0] - 1:
      q0, q1, denom = quats[-2], quats[-1], dt
    else:
      q0, q1, denom = quats[frame - 1], quats[frame + 1], 2.0 * dt
    delta = _quat_mul_wxyz(q1, _quat_conjugate_wxyz(q0))
    signs = np.where(delta[:, 0:1] < 0.0, -1.0, 1.0)
    delta = delta * signs
    xyz = delta[:, 1:4]
    w = np.clip(delta[:, 0], -1.0, 1.0)
    angle = 2.0 * np.arctan2(np.linalg.norm(xyz, axis=1), w)
    axis = np.zeros_like(xyz)
    norms = np.linalg.norm(xyz, axis=1)
    nonzero = norms > 1.0e-8
    axis[nonzero] = xyz[nonzero] / norms[nonzero, None]
    velocities[frame] = (axis * (angle[:, None] / denom)).astype(np.float32)
  return velocities


def emit_mjlab_motion_npz(
  source_path: str | Path,
  output_path: str | Path,
  *,
  output_fps: int = 50,
) -> Path:
  clip = resample_clip(load_omniretarget_clip(source_path), output_fps=output_fps)
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  data = mujoco.MjData(model)
  qpos = _to_mujoco_qpos(clip.qpos)
  qvel = _differentiate_qpos(model, qpos, 1.0 / float(output_fps))

  body_pos_w = np.zeros((qpos.shape[0], model.nbody - 1, 3), dtype=np.float32)
  body_quat_w = np.zeros((qpos.shape[0], model.nbody - 1, 4), dtype=np.float32)
  for frame in range(qpos.shape[0]):
    data.qpos[:] = qpos[frame]
    data.qvel[:] = qvel[frame]
    mujoco.mj_forward(model, data)
    body_pos_w[frame] = data.xpos[1:].astype(np.float32)
    body_quat_w[frame] = data.xquat[1:].astype(np.float32)

  dt = 1.0 / float(output_fps)
  body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0).astype(np.float32)
  body_ang_vel_w = _angular_velocity_from_quats(body_quat_w.astype(np.float64), dt)

  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    output_path,
    fps=np.array(output_fps, dtype=np.int32),
    joint_pos=clip.joint_pos.astype(np.float32),
    joint_vel=qvel[:, 6:].astype(np.float32),
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    body_lin_vel_w=body_lin_vel_w,
    body_ang_vel_w=body_ang_vel_w,
  )
  return output_path
