from __future__ import annotations

import json
import math
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
  path.parent.mkdir(parents=True, exist_ok=True)
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


def create_box_obj(
  path: Path,
  *,
  half_size: tuple[float, float, float] = (0.5, 0.25, 0.2),
  center: tuple[float, float, float] = (0.0, 0.0, 0.2),
  yaw: float = 0.0,
) -> Path:
  path.parent.mkdir(parents=True, exist_ok=True)
  hx, hy, hz = half_size
  cx, cy, cz = center
  rot = np.array(
    [
      [math.cos(yaw), -math.sin(yaw), 0.0],
      [math.sin(yaw), math.cos(yaw), 0.0],
      [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
  )
  local = np.array(
    [
      [-hx, -hy, -hz],
      [-hx, -hy, hz],
      [-hx, hy, -hz],
      [-hx, hy, hz],
      [hx, -hy, -hz],
      [hx, -hy, hz],
      [hx, hy, -hz],
      [hx, hy, hz],
    ],
    dtype=np.float64,
  )
  vertices = local @ rot.T + np.array([cx, cy, cz], dtype=np.float64)
  faces = [
    (1, 5, 7),
    (1, 7, 3),
    (2, 4, 8),
    (2, 8, 6),
    (1, 2, 6),
    (1, 6, 5),
    (3, 7, 8),
    (3, 8, 4),
    (1, 3, 4),
    (1, 4, 2),
    (5, 6, 8),
    (5, 8, 7),
  ]
  lines = [f"v {x:.8f} {y:.8f} {z:.8f}" for x, y, z in vertices]
  lines.extend(f"f {a} {b} {c}" for a, b, c in faces)
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return path


def create_omniretarget_terrain_urdf(
  path: Path,
  *,
  mesh_name: str = "box1.obj",
  scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Path:
  sx, sy, sz = scale
  path.parent.mkdir(parents=True, exist_ok=True)
  (path.parent / "box_models").mkdir(exist_ok=True)
  path.write_text(
    "\n".join(
      [
        '<?xml version="1.0"?>',
        '<robot name="multi_boxes">',
        '  <link name="box1_link">',
        "    <visual>",
        '      <geometry>',
        f'        <mesh filename="box_models/{mesh_name}" scale="{sx} {sy} {sz}"/>',
        '      </geometry>',
        "    </visual>",
        '    <collision name="box1">',
        '      <geometry>',
        f'        <mesh filename="box_models/{mesh_name}" scale="{sx} {sy} {sz}"/>',
        '      </geometry>',
        "    </collision>",
        "  </link>",
        "</robot>",
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
