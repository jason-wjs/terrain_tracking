from __future__ import annotations

import math
from pathlib import Path

import mujoco
import pytest

from terrain_tracking.scene.omniretarget_boxes import (
  load_omniretarget_terrain_boxes,
  make_omniretarget_boxes_spec_fn,
)
from tests.helpers import create_box_obj, create_omniretarget_terrain_urdf


def test_load_omniretarget_terrain_boxes_parses_collision_mesh_scale(
  tmp_path: Path,
) -> None:
  terrain_dir = tmp_path / "climb_00"
  create_box_obj(
    terrain_dir / "box_models" / "box1.obj",
    half_size=(0.5, 0.25, 0.2),
    center=(0.0, 0.0, 0.2),
  )
  urdf = create_omniretarget_terrain_urdf(
    terrain_dir / "multi_boxes_z_scale_1.0.urdf",
    scale=(2.0, 1.0, 0.5),
  )

  boxes = load_omniretarget_terrain_boxes(urdf)

  assert len(boxes) == 1
  assert boxes[0].name == "box1"
  assert boxes[0].mesh_file == (terrain_dir / "box_models" / "box1.obj").resolve()
  assert boxes[0].pos == pytest.approx((0.0, 0.0, 0.1))
  assert boxes[0].size == pytest.approx((1.0, 0.25, 0.1))


def test_load_omniretarget_terrain_boxes_preserves_yaw(
  tmp_path: Path,
) -> None:
  terrain_dir = tmp_path / "climb_00"
  create_box_obj(
    terrain_dir / "box_models" / "box1.obj",
    half_size=(0.8, 0.2, 0.3),
    yaw=math.pi / 6.0,
  )
  urdf = create_omniretarget_terrain_urdf(
    terrain_dir / "multi_boxes_z_scale_1.0.urdf"
  )

  box = load_omniretarget_terrain_boxes(urdf)[0]

  assert abs(abs(box.yaw) - math.pi / 6.0) < 1.0e-5
  assert sorted(box.size[:2]) == pytest.approx([0.2, 0.8])
  assert box.size[2] == pytest.approx(0.3)


def test_make_omniretarget_boxes_spec_fn_compiles_box_geoms(
  tmp_path: Path,
) -> None:
  terrain_dir = tmp_path / "climb_00"
  create_box_obj(terrain_dir / "box_models" / "box1.obj")
  urdf = create_omniretarget_terrain_urdf(
    terrain_dir / "multi_boxes_z_scale_1.0.urdf"
  )
  spec = mujoco.MjSpec()

  make_omniretarget_boxes_spec_fn(urdf)(spec)
  model = spec.compile()

  geom_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    for geom_id in range(model.ngeom)
  ]
  assert "omniretarget_ground" in geom_names
  assert "omniretarget_box1" in geom_names
  assert all(geom_type == mujoco.mjtGeom.mjGEOM_BOX for geom_type in model.geom_type)


def test_make_omniretarget_boxes_spec_fn_uses_one_shared_terrain_when_spacing_is_zero(
  tmp_path: Path,
) -> None:
  terrain_dir = tmp_path / "climb_00"
  create_box_obj(terrain_dir / "box_models" / "box1.obj")
  urdf = create_omniretarget_terrain_urdf(
    terrain_dir / "multi_boxes_z_scale_1.0.urdf"
  )
  spec = mujoco.MjSpec()

  make_omniretarget_boxes_spec_fn(urdf, num_envs=4, env_spacing=0.0)(spec)

  assert len(spec.worldbody.bodies) == 1
  body = spec.worldbody.bodies[0]
  assert body.name == "omniretarget_terrain"
  assert [geom.name for geom in body.geoms] == [
    "omniretarget_ground",
    "omniretarget_box1",
  ]
