from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import mujoco
import numpy as np
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.terrains import SubTerrainCfg, TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

from terrain_tracking.runtime.pair_manifest import PairManifest
from terrain_tracking.runtime.terrain_collision import TerrainCollisionManifest
from terrain_tracking.scene.heightfield_spec import make_heightfield_spec_fn
from terrain_tracking.scene.omniretarget_boxes import make_omniretarget_boxes_spec_fn
from terrain_tracking.scene.paired_mesh_spec import make_paired_mesh_spec_fn
from terrain_tracking.scene.primitive_box_terrain import (
  build_primitive_box_tile,
  make_primitive_box_tile_spec_fn,
)

CollisionBackend = Literal["primitive_boxes", "hfield", "mesh", "omniretarget_boxes"]
_COLLISION_BACKENDS: tuple[str, ...] = (
  "primitive_boxes",
  "hfield",
  "mesh",
  "omniretarget_boxes",
)


@dataclass(kw_only=True)
class PairPrimitiveBoxTerrainCfg(SubTerrainCfg):
  collision_file: str

  def function(
    self,
    difficulty: float,
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
  ) -> TerrainOutput:
    del difficulty, rng
    manifest = TerrainCollisionManifest.load(self.collision_file)
    body = spec.body("terrain")
    local_offset = (0.5 * self.size[0], 0.5 * self.size[1], 0.0)
    tile = make_primitive_box_tile_spec_fn(
      manifest,
      local_offset=local_offset,
    )(spec, body)
    return TerrainOutput(
      origin=np.array(local_offset),
      geometries=[
        TerrainGeometry(geom=geom)
        for geom in body.geoms[-len(tile.boxes) :]
      ],
    )


def _set_min_sim_contact_buffers(
  cfg: ManagerBasedRlEnvCfg,
  *,
  nconmax: int,
  njmax: int,
) -> None:
  if cfg.sim.nconmax is None or cfg.sim.nconmax < nconmax:
    cfg.sim.nconmax = nconmax
  if cfg.sim.njmax is None or cfg.sim.njmax < njmax:
    cfg.sim.njmax = njmax


def _parse_collision_backend(value: str) -> CollisionBackend:
  if value not in _COLLISION_BACKENDS:
    raise ValueError(
      "collision_backend must be one of: primitive_boxes, hfield, mesh, "
      "omniretarget_boxes; "
      f"got {value!r}"
    )
  return cast(CollisionBackend, value)


def _apply_motion_file(cfg: ManagerBasedRlEnvCfg, pair: PairManifest) -> None:
  motion_cfg = cfg.commands["motion"]
  if not isinstance(motion_cfg, MotionCommandCfg):
    raise TypeError("Expected cfg.commands['motion'] to be a MotionCommandCfg")
  motion_cfg.motion_file = str(pair.motion_file)


def _apply_primitive_box_generator(
  cfg: ManagerBasedRlEnvCfg,
  pair: PairManifest,
) -> bool:
  if pair.terrain_collision_file is None:
    return False

  collision = TerrainCollisionManifest.load(pair.terrain_collision_file)
  tile = build_primitive_box_tile(collision)
  if cfg.scene.terrain is None:
    cfg.scene.terrain = TerrainEntityCfg()
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=tile.footprint,
    border_width=0.0,
    border_height=0.0,
    num_rows=1,
    num_cols=1,
    curriculum=False,
    sub_terrains={
      "pair": PairPrimitiveBoxTerrainCfg(
        size=tile.footprint,
        collision_file=str(pair.terrain_collision_file),
      ),
    },
  )
  cfg.scene.env_spacing = 0.0
  return True


def _make_pair_spec_fn(
  cfg: ManagerBasedRlEnvCfg,
  pair: PairManifest,
  collision_backend: CollisionBackend,
) -> Callable[[mujoco.MjSpec], None]:
  def pair_spec_fn(spec: mujoco.MjSpec) -> None:
    if collision_backend == "omniretarget_boxes":
      make_omniretarget_boxes_spec_fn(
        pair.terrain_file,
        motion_file=pair.motion_file,
        num_envs=cfg.scene.num_envs,
        env_spacing=cfg.scene.env_spacing,
      )(spec)
    elif pair.terrain_collision_file is not None and collision_backend == "hfield":
      make_heightfield_spec_fn(
        TerrainCollisionManifest.load(pair.terrain_collision_file),
        num_envs=cfg.scene.num_envs,
        env_spacing=cfg.scene.env_spacing,
      )(spec)
    else:
      make_paired_mesh_spec_fn(
        pair,
        num_envs=cfg.scene.num_envs,
        env_spacing=cfg.scene.env_spacing,
      )(spec)

  return pair_spec_fn


def apply_pair_manifest_to_env_cfg(
  cfg: ManagerBasedRlEnvCfg,
  manifest: str | Path | PairManifest,
  *,
  collision_backend: str = "primitive_boxes",
) -> PairManifest:
  backend = _parse_collision_backend(collision_backend)

  pair = manifest if isinstance(manifest, PairManifest) else PairManifest.load(manifest)

  _apply_motion_file(cfg, pair)
  if pair.terrain_collision_file is not None or backend == "omniretarget_boxes":
    _set_min_sim_contact_buffers(cfg, nconmax=256, njmax=512)
  if backend == "omniretarget_boxes":
    cfg.scene.env_spacing = 0.0

  if backend == "primitive_boxes" and _apply_primitive_box_generator(cfg, pair):
    return pair

  if backend == "hfield" and pair.terrain_collision_file is None:
    raise ValueError("collision_backend='hfield' requires terrain_collision_file")

  existing_spec_fn = cfg.scene.spec_fn
  pair_spec_fn = _make_pair_spec_fn(cfg, pair, backend)

  if existing_spec_fn is None:
    cfg.scene.spec_fn = pair_spec_fn
  else:

    def chained_spec_fn(spec) -> None:
      existing_spec_fn(spec)
      pair_spec_fn(spec)

    cfg.scene.spec_fn = chained_spec_fn

  return pair
