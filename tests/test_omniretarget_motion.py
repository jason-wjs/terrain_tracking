from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from terrain_tracking.omniretarget_motion import (
  emit_mjlab_motion_npz,
  load_omniretarget_clip,
  resample_clip,
)
from tests.helpers import create_omniretarget_qpos_clip


def test_load_omniretarget_clip_parses_qpos_layout(tmp_path: Path) -> None:
  clip_path = create_omniretarget_qpos_clip(tmp_path / "clip.npz", num_frames=4)

  clip = load_omniretarget_clip(clip_path)

  assert clip.fps == 30
  assert clip.root_quat_wxyz.shape == (4, 4)
  assert clip.root_pos_w.shape == (4, 3)
  assert clip.joint_pos.shape == (4, 29)
  assert clip.root_quat_wxyz[0].tolist() == [1.0, 0.0, 0.0, 0.0]
  assert clip.root_pos_w[2].tolist() == pytest.approx([0.2, -0.1, 0.84])
  assert clip.joint_pos[3, 0] == pytest.approx(0.03)


def test_load_omniretarget_clip_rejects_non_g1_qpos_shape(tmp_path: Path) -> None:
  path = tmp_path / "bad.npz"
  np.savez(path, qpos=np.zeros((4, 35), dtype=np.float32), fps=np.array(30))

  with pytest.raises(ValueError, match="qpos"):
    load_omniretarget_clip(path)


def test_resample_clip_outputs_control_rate_frames(tmp_path: Path) -> None:
  clip = load_omniretarget_clip(
    create_omniretarget_qpos_clip(tmp_path / "clip.npz", num_frames=4, fps=30)
  )

  resampled = resample_clip(clip, output_fps=50)

  assert resampled.fps == 50
  assert resampled.qpos.shape == (6, 36)
  assert np.all(np.diff(resampled.times) > 0.0)
  assert resampled.root_pos_w[0].tolist() == pytest.approx(clip.root_pos_w[0])
  assert resampled.root_pos_w[-1].tolist() == pytest.approx(clip.root_pos_w[-1])
  assert np.linalg.norm(resampled.root_quat_wxyz, axis=1).tolist() == pytest.approx(
    [1.0] * resampled.qpos.shape[0]
  )


def test_emit_mjlab_motion_npz_writes_tracking_schema(tmp_path: Path) -> None:
  clip_path = create_omniretarget_qpos_clip(tmp_path / "clip.npz", num_frames=4)
  output_path = tmp_path / "motion.npz"

  emit_mjlab_motion_npz(clip_path, output_path, output_fps=50)

  data = np.load(output_path)
  assert set(data.files) == {
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  }
  assert int(np.asarray(data["fps"]).item()) == 50
  assert data["joint_pos"].shape == (6, 29)
  assert data["joint_vel"].shape == (6, 29)
  assert data["body_pos_w"].shape[0] == 6
  assert data["body_pos_w"].shape[-1] == 3
  assert data["body_quat_w"].shape[-1] == 4
  assert data["body_lin_vel_w"].shape == data["body_pos_w"].shape
  assert data["body_ang_vel_w"].shape == data["body_pos_w"].shape
