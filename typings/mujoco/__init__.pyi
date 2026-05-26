from __future__ import annotations

from typing import Any, Sequence

class _GeomType:
  mjGEOM_BOX: Any
  mjGEOM_HFIELD: Any
  mjGEOM_MESH: Any
  mjGEOM_PLANE: Any


class _ObjType:
  mjOBJ_GEOM: Any


mjtGeom: _GeomType
mjtObj: _ObjType


class MjsGeom:
  name: str
  type: Any
  meshname: str
  mass: float
  contype: int
  conaffinity: int


class MjsBody:
  name: str
  pos: Sequence[float]
  geoms: list[MjsGeom]
  bodies: list[MjsBody]

  def add_body(
    self,
    *,
    name: str,
    pos: Sequence[float] | None = None,
  ) -> MjsBody: ...

  def add_geom(
    self,
    *,
    name: str,
    type: Any,
    pos: Sequence[float] | None = None,
    size: Sequence[float] | None = None,
    quat: Sequence[float] | None = None,
    meshname: str | None = None,
    hfieldname: str | None = None,
    contype: int | None = None,
    conaffinity: int | None = None,
  ) -> MjsGeom: ...


class MjSpec:
  nconmax: int
  njmax: int
  worldbody: MjsBody
  hfields: list[Any]
  meshes: list[Any]

  def body(self, name: str) -> MjsBody: ...
  def compile(self) -> MjModel: ...
  def add_hfield(
    self,
    *,
    name: str,
    nrow: int,
    ncol: int,
    size: Sequence[float],
    userdata: Sequence[float],
  ) -> Any: ...
  def add_mesh(
    self,
    *,
    name: str,
    uservert: Sequence[float],
    userface: Sequence[int],
  ) -> Any: ...


class MjModel:
  nv: int
  nbody: int
  ngeom: int
  nhfield: int
  geom_type: Any
  geom_contype: Any
  geom_conaffinity: Any

  @classmethod
  def from_xml_path(cls, path: str) -> MjModel: ...


class MjData:
  qpos: Any
  qvel: Any
  xpos: Any
  xquat: Any

  def __init__(self, model: MjModel) -> None: ...


def mj_differentiatePos(
  model: MjModel,
  qvel: Any,
  dt: float,
  qpos1: Any,
  qpos2: Any,
) -> None: ...


def mj_forward(model: MjModel, data: MjData) -> None: ...
def mj_id2name(model: MjModel, objtype: Any, objid: int) -> str | None: ...
def mj_name2id(model: MjModel, objtype: Any, name: str) -> int: ...
