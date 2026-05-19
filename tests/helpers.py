from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def create_motion_clip(
  path: Path,
  *,
  num_frames: int = 8,
  num_joints: int = 29,
  num_bodies: int = 30,
) -> Path:
  joint_pos = np.zeros((num_frames, num_joints), dtype=np.float32)
  joint_vel = np.zeros((num_frames, num_joints), dtype=np.float32)

  body_pos_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
  body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
  body_lin_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
  body_ang_vel_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)

  body_quat_w[..., 0] = 1.0

  for frame in range(num_frames):
    body_pos_w[frame, :, 0] = np.linspace(0.0, 0.5, num_bodies, dtype=np.float32)
    body_pos_w[frame, :, 1] = np.linspace(-0.2, 0.2, num_bodies, dtype=np.float32)
    body_pos_w[frame, :, 2] = 0.9 + np.linspace(
      0.0, 0.1, num_bodies, dtype=np.float32
    )
    body_pos_w[frame, :, 0] += 0.02 * frame

  body_lin_vel_w[1:] = np.diff(body_pos_w, axis=0)

  np.savez(
    path,
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    body_lin_vel_w=body_lin_vel_w,
    body_ang_vel_w=body_ang_vel_w,
  )
  return path


def create_omniretarget_qpos_clip(
  path: Path,
  *,
  num_frames: int = 4,
  fps: int = 30,
) -> Path:
  qpos = np.zeros((num_frames, 36), dtype=np.float64)
  qpos[:, 0] = 1.0
  for frame in range(num_frames):
    qpos[frame, 4:7] = (0.1 * frame, -0.05 * frame, 0.8 + 0.02 * frame)
    qpos[frame, 7:] = np.linspace(
      0.0,
      0.28,
      29,
      dtype=np.float64,
    ) + 0.01 * frame
  np.savez(path, qpos=qpos, fps=np.array(fps, dtype=np.int32))
  return path


def create_quad_obj(path: Path) -> Path:
  path.write_text(
    "\n".join(
      [
        "v 0.0 0.0 0.0",
        "v 1.0 0.0 0.0",
        "v 1.0 1.0 0.0",
        "v 0.0 1.0 0.0",
        "f 1 2 3",
        "f 1 3 4",
      ]
    )
    + "\n",
    encoding="utf-8",
  )
  return path


def create_ramp_obj(path: Path) -> Path:
  path.write_text(
    "\n".join(
      [
        "v 0.0 0.0 0.0",
        "v 1.0 0.0 0.0",
        "v 1.0 1.0 0.1",
        "v 0.0 1.0 0.0",
        "f 1 2 3",
        "f 1 3 4",
      ]
    )
    + "\n",
    encoding="utf-8",
  )
  return path


def create_pair_manifest(
  path: Path,
  *,
  motion_file: str,
  terrain_file: str,
  terrain_translation: tuple[float, float, float] | None = None,
  terrain_quat_xyzw: tuple[float, float, float, float] | None = None,
  terrain_scale: tuple[float, float, float] | None = None,
) -> Path:
  payload: dict[str, object] = {
    "motion_file": motion_file,
    "terrain_file": terrain_file,
  }
  if terrain_translation is not None:
    payload["terrain_translation"] = list(terrain_translation)
  if terrain_quat_xyzw is not None:
    payload["terrain_quat_xyzw"] = list(terrain_quat_xyzw)
  if terrain_scale is not None:
    payload["terrain_scale"] = list(terrain_scale)
  path.write_text(json.dumps(payload), encoding="utf-8")
  return path


def create_heightfield_collision_manifest(path: Path) -> Path:
  np.save(
    path.parent / "terrain_hf.npy",
    np.array([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32),
  )
  path.write_text(
    json.dumps(
      {
        "schema_version": 1,
        "terrain_name": "test_hfield",
        "collision": {
          "type": "heightfield",
          "hf_file": "terrain_hf.npy",
          "min_point": [-0.2, -0.2],
          "dx": 0.4,
          "base_z": -0.4,
          "xy_scale": 1.0,
          "height_scale": 1.0,
        },
      }
    ),
    encoding="utf-8",
  )
  return path
