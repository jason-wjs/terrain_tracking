from __future__ import annotations

import mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg


def _disable_default_plane_collision(spec: mujoco.MjSpec) -> None:
  for body in spec.worldbody.bodies:
    if body.name != "terrain":
      continue
    for geom in body.geoms:
      if geom.name == "terrain" and geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
        geom.contype = 0
        geom.conaffinity = 0


def unitree_g1_blind_terrain_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=has_state_estimation,
    play=play,
  )

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  cfg.scene.spec_fn = _disable_default_plane_collision

  if not play:
    motion_cmd.pose_range = {
      "x": (0.0, 0.0),
      "y": (0.0, 0.0),
      "z": (0.0, 0.0),
      "roll": (0.0, 0.0),
      "pitch": (0.0, 0.0),
      "yaw": (0.0, 0.0),
    }
    motion_cmd.velocity_range = {
      "x": (0.0, 0.0),
      "y": (0.0, 0.0),
      "z": (0.0, 0.0),
      "roll": (0.0, 0.0),
      "pitch": (0.0, 0.0),
      "yaw": (0.0, 0.0),
    }
    motion_cmd.joint_position_range = (0.0, 0.0)
    cfg.events.pop("push_robot", None)

  return cfg
