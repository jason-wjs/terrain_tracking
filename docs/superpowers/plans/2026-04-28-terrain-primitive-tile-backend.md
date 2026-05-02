# Terrain Primitive Tile Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the default PARC terrain collision path with primitive box collision and stop duplicating terrain geoms per env by routing single-pair terrain through a shared mjlab terrain tile.

**Architecture:** terrain_tracking will keep pair manifests as the user-facing contract, but `terrain_collision.json` will be converted into a tile-level primitive box terrain. For single-pair training, the scene will contain one terrain tile and all envs will share that tile origin across separate MuJoCo-Warp worlds. Hfield and OBJ mesh collision remain available as explicit/debug fallbacks, not the default training backend.

**Tech Stack:** Python, NumPy, MuJoCo `MjSpec`, mjlab `TerrainGeneratorCfg`/`SubTerrainCfg`, pytest, uv.

---

## File Structure

- Create `src/terrain_tracking/scene/primitive_box_terrain.py`
  - Load a `TerrainCollisionManifest`.
  - Convert PARC cell-centered hf data into merged primitive box descriptors.
  - Provide a mjlab `SubTerrainCfg` that emits MuJoCo `box` geoms for one tile.
  - Return diagnostics such as unique heights, nonzero cells, and merged box count.

- Modify `src/terrain_tracking/runtime/apply_pair.py`
  - Choose primitive box tile backend by default when `terrain_collision_file` exists.
  - Configure `cfg.scene.terrain.terrain_type = "generator"` and install a one-tile `TerrainGeneratorCfg`.
  - Keep hfield backend selectable for debug.

- Modify `src/terrain_tracking/tasks/blind_terrain_tracking/scripts/train.py`
  - Add a frontend flag for collision backend, defaulting to `primitive_boxes`.

- Modify `src/terrain_tracking/tasks/blind_terrain_tracking/scripts/play.py`
  - Add the same backend flag so play and train load identical collision when desired.

- Modify `src/terrain_tracking/tasks/blind_terrain_tracking/scripts/common.py`
  - Thread backend selection through `build_paired_env()`.

- Create `tests/test_primitive_box_terrain.py`
  - Unit-test hf-to-box conversion, rectangle merge, coordinate mapping, and MuJoCo compilation.

- Modify `tests/test_env_smoke.py`
  - Assert collision manifests now default to primitive box terrain and do not create one terrain geom per env.

- Modify `docs/terrain-collision-contract.md`
  - Document that `terrain_hf.npy` is source terrain data and `primitive_boxes` is the default collision backend for PARC platform/block training.

---

### Task 1: Add Primitive Box Conversion Unit Tests

**Files:**
- Create: `tests/test_primitive_box_terrain.py`
- Create later in Task 2: `src/terrain_tracking/scene/primitive_box_terrain.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_primitive_box_terrain.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from terrain_tracking.runtime.terrain_collision import TerrainCollisionManifest
from terrain_tracking.scene.primitive_box_terrain import (
  build_primitive_box_tile,
  make_primitive_box_tile_spec_fn,
)


def _write_collision(tmp_path: Path, hf: np.ndarray) -> TerrainCollisionManifest:
  np.save(tmp_path / "terrain_hf.npy", hf.astype(np.float32))
  manifest_path = tmp_path / "terrain_collision.json"
  manifest_path.write_text(
    json.dumps(
      {
        "schema_version": 1,
        "terrain_name": "demo",
        "collision": {
          "type": "heightfield",
          "hf_file": "terrain_hf.npy",
          "min_point": [-0.8, -0.8],
          "dx": 0.4,
          "base_z": -0.4,
          "xy_scale": 0.5,
          "height_scale": 0.5,
        },
      }
    ),
    encoding="utf-8",
  )
  return TerrainCollisionManifest.load(manifest_path)


def test_build_primitive_box_tile_merges_constant_height_rectangles(tmp_path: Path) -> None:
  hf = np.array(
    [
      [0.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 1.0, 0.0],
      [0.0, 1.0, 1.0, 0.0],
    ],
    dtype=np.float32,
  )
  manifest = _write_collision(tmp_path, hf)

  tile = build_primitive_box_tile(manifest)

  assert tile.diagnostics.hf_shape == (3, 4)
  assert tile.diagnostics.unique_height_count == 2
  assert tile.diagnostics.nonzero_cell_count == 4
  assert tile.diagnostics.box_count == 2
  assert tile.boxes[0].name == "terrain_base"
  assert tile.boxes[1].name == "terrain_h0_rect0"
  assert tile.boxes[0].size == pytest.approx((0.3, 0.4, 0.2))
  assert tile.boxes[0].pos == pytest.approx((-0.2, -0.1, -0.2))
  assert tile.boxes[1].size == pytest.approx((0.2, 0.2, 0.25))
  assert tile.boxes[1].pos == pytest.approx((-0.1, -0.1, 0.25))
  assert tile.footprint == pytest.approx((0.6, 0.8))


def test_make_primitive_box_tile_spec_fn_compiles_one_tile_not_per_env(tmp_path: Path) -> None:
  hf = np.array(
    [
      [0.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 1.0, 0.0],
      [0.0, 1.0, 1.0, 0.0],
    ],
    dtype=np.float32,
  )
  manifest = _write_collision(tmp_path, hf)
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="terrain")

  make_primitive_box_tile_spec_fn(manifest)(spec, body)
  model = spec.compile()

  assert model.ngeom == 2
  assert model.geom_type[0] == mujoco.mjtGeom.mjGEOM_BOX
  assert model.geom_type[1] == mujoco.mjtGeom.mjGEOM_BOX
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest -q tests/test_primitive_box_terrain.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'terrain_tracking.scene.primitive_box_terrain'`.

---

### Task 2: Implement Primitive Box Tile Conversion

**Files:**
- Create: `src/terrain_tracking/scene/primitive_box_terrain.py`
- Modify: `src/terrain_tracking/scene/__init__.py`
- Test: `tests/test_primitive_box_terrain.py`

- [ ] **Step 1: Add primitive box terrain implementation**

Create `src/terrain_tracking/scene/primitive_box_terrain.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np

from terrain_tracking.runtime.terrain_collision import TerrainCollisionManifest


@dataclass(frozen=True)
class PrimitiveBox:
  name: str
  pos: tuple[float, float, float]
  size: tuple[float, float, float]


@dataclass(frozen=True)
class PrimitiveBoxDiagnostics:
  hf_shape: tuple[int, int]
  unique_height_count: int
  nonzero_cell_count: int
  box_count: int


@dataclass(frozen=True)
class PrimitiveBoxTile:
  boxes: tuple[PrimitiveBox, ...]
  diagnostics: PrimitiveBoxDiagnostics
  footprint: tuple[float, float]


def _load_scaled_hf(manifest: TerrainCollisionManifest) -> np.ndarray:
  hf = np.load(manifest.hf_file).astype(np.float32)
  if hf.ndim != 2:
    raise ValueError(f"heightfield must be 2D, got shape {hf.shape}")
  return hf * np.float32(manifest.height_scale)


def _greedy_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
  used = np.zeros(mask.shape, dtype=bool)
  rects: list[tuple[int, int, int, int]] = []
  nx, ny = mask.shape
  for i0 in range(nx):
    for j0 in range(ny):
      if used[i0, j0] or not mask[i0, j0]:
        continue
      j1 = j0
      while j1 + 1 < ny and mask[i0, j1 + 1] and not used[i0, j1 + 1]:
        j1 += 1
      i1 = i0
      while i1 + 1 < nx:
        row = mask[i1 + 1, j0 : j1 + 1]
        row_used = used[i1 + 1, j0 : j1 + 1]
        if not bool(np.all(row & ~row_used)):
          break
        i1 += 1
      used[i0 : i1 + 1, j0 : j1 + 1] = True
      rects.append((i0, i1, j0, j1))
  return rects


def _rect_to_box(
  *,
  name: str,
  i0: int,
  i1: int,
  j0: int,
  j1: int,
  height: float,
  manifest: TerrainCollisionManifest,
) -> PrimitiveBox:
  cell = float(manifest.dx * manifest.xy_scale)
  min_x = float(manifest.min_point[0] * manifest.xy_scale)
  min_y = float(manifest.min_point[1] * manifest.xy_scale)
  sx = 0.5 * float(i1 - i0 + 1) * cell
  sy = 0.5 * float(j1 - j0 + 1) * cell
  sz = 0.5 * float(height)
  cx = min_x + float(i0 + i1) * 0.5 * cell
  cy = min_y + float(j0 + j1) * 0.5 * cell
  cz = sz
  return PrimitiveBox(name=name, pos=(cx, cy, cz), size=(sx, sy, sz))


def _shift_box(box: PrimitiveBox, offset: tuple[float, float, float]) -> PrimitiveBox:
  return PrimitiveBox(
    name=box.name,
    pos=(
      box.pos[0] + offset[0],
      box.pos[1] + offset[1],
      box.pos[2] + offset[2],
    ),
    size=box.size,
  )


def build_primitive_box_tile(
  manifest: TerrainCollisionManifest,
  *,
  height_epsilon: float = 1.0e-5,
  max_boxes: int = 128,
) -> PrimitiveBoxTile:
  hf = _load_scaled_hf(manifest)
  nx, ny = hf.shape
  footprint = (
    float(nx * manifest.dx * manifest.xy_scale),
    float(ny * manifest.dx * manifest.xy_scale),
  )
  cell = float(manifest.dx * manifest.xy_scale)
  min_x = float(manifest.min_point[0] * manifest.xy_scale)
  min_y = float(manifest.min_point[1] * manifest.xy_scale)
  center_x = min_x + 0.5 * float(nx - 1) * cell
  center_y = min_y + 0.5 * float(ny - 1) * cell
  base_depth = max(abs(float(manifest.base_z)), 1.0e-6)
  base = PrimitiveBox(
    name="terrain_base",
    pos=(center_x, center_y, -0.5 * base_depth),
    size=(0.5 * footprint[0], 0.5 * footprint[1], 0.5 * base_depth),
  )

  heights = np.unique(hf[hf > height_epsilon])
  boxes = [base]
  for height_idx, height in enumerate(heights):
    mask = np.isclose(hf, height, atol=height_epsilon)
    for rect_idx, (i0, i1, j0, j1) in enumerate(_greedy_rectangles(mask)):
      boxes.append(
        _rect_to_box(
          name=f"terrain_h{height_idx}_rect{rect_idx}",
          i0=i0,
          i1=i1,
          j0=j0,
          j1=j1,
          height=float(height),
          manifest=manifest,
        )
      )

  if len(boxes) > max_boxes:
    raise ValueError(
      "primitive box terrain is too complex: "
      f"box_count={len(boxes)}, max_boxes={max_boxes}, hf_shape={hf.shape}"
    )

  diagnostics = PrimitiveBoxDiagnostics(
    hf_shape=(int(nx), int(ny)),
    unique_height_count=int(len(np.unique(hf))),
    nonzero_cell_count=int(np.count_nonzero(hf > height_epsilon)),
    box_count=len(boxes),
  )
  return PrimitiveBoxTile(boxes=tuple(boxes), diagnostics=diagnostics, footprint=footprint)


def make_primitive_box_tile_spec_fn(
  manifest: TerrainCollisionManifest,
  *,
  max_boxes: int = 128,
  local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Callable[[mujoco.MjSpec, mujoco.MjsBody], PrimitiveBoxTile]:
  tile = build_primitive_box_tile(manifest, max_boxes=max_boxes)
  shifted_boxes = tuple(_shift_box(box, local_offset) for box in tile.boxes)

  def spec_fn(_spec: mujoco.MjSpec, body: mujoco.MjsBody) -> PrimitiveBoxTile:
    for box in shifted_boxes:
      geom = body.add_geom(
        name=box.name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=box.pos,
        size=box.size,
        contype=1,
        conaffinity=1,
      )
      geom.mass = 0
    return tile

  return spec_fn
```

- [ ] **Step 2: Export new helpers**

Modify `src/terrain_tracking/scene/__init__.py`:

```python
from .primitive_box_terrain import (
  PrimitiveBox,
  PrimitiveBoxDiagnostics,
  PrimitiveBoxTile,
  build_primitive_box_tile,
  make_primitive_box_tile_spec_fn,
)
```

Add the same names to `__all__`.

- [ ] **Step 3: Run primitive box tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest -q tests/test_primitive_box_terrain.py
```

Expected: PASS.

---

### Task 3: Route Collision Manifests To A Shared Generator Tile

**Files:**
- Modify: `src/terrain_tracking/runtime/apply_pair.py`
- Test: `tests/test_env_smoke.py`

- [ ] **Step 1: Write failing smoke test for single shared terrain tile**

Append to `tests/test_env_smoke.py`:

```python
def test_hfield_manifest_defaults_to_one_shared_primitive_box_tile(
  tmp_path: Path,
) -> None:
  collision_path = create_heightfield_collision_manifest(
    tmp_path / "terrain_collision.json"
  )
  bundle_dir = convert_pair(
    ConvertPairConfig(
      motion_file=create_motion_clip(tmp_path / "motion.npz"),
      terrain_file=create_ramp_obj(tmp_path / "terrain.obj"),
      terrain_collision_file=collision_path,
      output_dir=tmp_path / "converted",
      sample_name="smoke_pair",
    )
  )

  env, _agent_cfg = build_paired_env(
    "TT-Tracking-TerrainBlind-Unitree-G1",
    str(bundle_dir / "pair.json"),
    play=True,
    device="cpu",
    num_envs=4,
    no_terminations=True,
  )
  try:
    model = env.unwrapped.sim.mj_model
    terrain_geoms = [
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
      for geom_id in range(model.ngeom)
    ]
    primitive_terrain_geoms = [
      name for name in terrain_geoms if name and name.startswith("terrain_")
    ]

    assert model.nhfield == 0
    assert len(primitive_terrain_geoms) > 1
    assert len(primitive_terrain_geoms) < 10
    assert torch.allclose(env.unwrapped.scene.env_origins, torch.zeros_like(env.unwrapped.scene.env_origins))
  finally:
    env.close()
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest -q tests/test_env_smoke.py::test_hfield_manifest_defaults_to_one_shared_primitive_box_tile
```

Expected: FAIL because the current manifest path uses hfield and creates one terrain geom per env.

- [ ] **Step 3: Implement shared generator tile in `apply_pair.py`**

Modify `src/terrain_tracking/runtime/apply_pair.py`:

```python
from dataclasses import dataclass

from mjlab.terrains import SubTerrainCfg, TerrainGeneratorCfg, TerrainGeometry, TerrainOutput
import mujoco
import numpy as np

from terrain_tracking.scene.primitive_box_terrain import (
  build_primitive_box_tile,
  make_primitive_box_tile_spec_fn,
)
```

Add:

```python
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
      geometries=[TerrainGeometry(geom=geom) for geom in body.geoms[-len(tile.boxes) :]],
    )
```

Then replace the default collision-manifest path:

```python
if pair.terrain_collision_file is not None:
  collision = TerrainCollisionManifest.load(pair.terrain_collision_file)
  tile = build_primitive_box_tile(collision)
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
      )
    },
  )
  cfg.scene.env_spacing = 0.0
```

Keep the existing hfield `spec_fn` path behind a backend flag in Task 4. For this task, the default path should no longer call `make_heightfield_spec_fn()` when `terrain_collision_file` is present.

- [ ] **Step 4: Run smoke test**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest -q tests/test_env_smoke.py::test_hfield_manifest_defaults_to_one_shared_primitive_box_tile
```

Expected: PASS.

---

### Task 4: Add Explicit Collision Backend Selection

**Files:**
- Modify: `src/terrain_tracking/runtime/apply_pair.py`
- Modify: `src/terrain_tracking/tasks/blind_terrain_tracking/scripts/common.py`
- Modify: `src/terrain_tracking/tasks/blind_terrain_tracking/scripts/train.py`
- Modify: `src/terrain_tracking/tasks/blind_terrain_tracking/scripts/play.py`
- Test: `tests/test_cli_help.py`
- Test: `tests/test_env_smoke.py`

- [ ] **Step 1: Add backend parameter to runtime API**

Change signature in `apply_pair.py`:

```python
def apply_pair_manifest_to_env_cfg(
  cfg: ManagerBasedRlEnvCfg,
  manifest: str | Path | PairManifest,
  *,
  collision_backend: str = "primitive_boxes",
) -> PairManifest:
```

Allowed values:

```python
if collision_backend not in {"primitive_boxes", "hfield", "mesh"}:
  raise ValueError(
    "collision_backend must be one of: primitive_boxes, hfield, mesh; "
    f"got {collision_backend!r}"
  )
```

Behavior:

- `primitive_boxes`: use shared generator tile.
- `hfield`: use existing `make_heightfield_spec_fn()` path for debug.
- `mesh`: ignore `terrain_collision_file` and use existing `make_paired_mesh_spec_fn()` fallback.

- [ ] **Step 2: Thread backend through common builder**

Modify `build_paired_env()` in `common.py`:

```python
def build_paired_env(
  task_id: str,
  pair_manifest: str,
  *,
  play: bool,
  device: str,
  num_envs: int | None = None,
  env_spacing: float | None = None,
  render_mode: str | None = None,
  no_terminations: bool = False,
  collision_backend: str = "primitive_boxes",
) -> tuple[ManagerBasedRlEnv, RslRlBaseRunnerCfg]:
```

Call:

```python
apply_pair_manifest_to_env_cfg(
  env_cfg,
  pair_manifest,
  collision_backend=collision_backend,
)
```

- [ ] **Step 3: Add train CLI field**

Modify `FrontendConfig` in `train.py`:

```python
@dataclass(frozen=True)
class FrontendConfig:
  pair_manifest: str
  task: str = DEFAULT_TASK_ID
  collision_backend: str = "primitive_boxes"
```

Call:

```python
apply_pair_manifest_to_env_cfg(
  train_cfg.env,
  frontend.pair_manifest,
  collision_backend=frontend.collision_backend,
)
```

- [ ] **Step 4: Add play CLI field**

Modify `PlayConfig` in `play.py`:

```python
collision_backend: str = "primitive_boxes"
```

Pass it to `build_paired_env()`.

- [ ] **Step 5: Add CLI help tests**

Append to `tests/test_cli_help.py`:

```python
def test_train_module_help_mentions_collision_backend() -> None:
  proc = subprocess.run(
    [
      sys.executable,
      "-m",
      "terrain_tracking.tasks.blind_terrain_tracking.scripts.train",
      "--help",
    ],
    check=True,
    text=True,
    capture_output=True,
  )
  assert "--collision-backend" in proc.stdout


def test_play_module_help_mentions_collision_backend() -> None:
  proc = subprocess.run(
    [
      sys.executable,
      "-m",
      "terrain_tracking.tasks.blind_terrain_tracking.scripts.play",
      "--help",
    ],
    check=True,
    text=True,
    capture_output=True,
  )
  assert "--collision-backend" in proc.stdout
```

- [ ] **Step 6: Run targeted tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest -q tests/test_cli_help.py tests/test_env_smoke.py
```

Expected: PASS.

---

### Task 5: Add Zero-Action Regression Diagnostic

**Files:**
- Create: `scripts/debug_zero_action_collision_backend.sh`
- Create: `src/terrain_tracking/tasks/blind_terrain_tracking/scripts/debug_zero_action.py`

- [ ] **Step 1: Add debug script**

Create `src/terrain_tracking/tasks/blind_terrain_tracking/scripts/debug_zero_action.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
import tyro

from mjlab.tasks.tracking.mdp import MotionCommandCfg

from terrain_tracking.tasks.blind_terrain_tracking.scripts.common import (
  DEFAULT_TASK_ID,
  build_paired_env,
  configure_runtime,
)


@dataclass(frozen=True)
class DebugConfig:
  pair_manifest: str
  task: str = DEFAULT_TASK_ID
  device: str | None = "cpu"
  num_envs: int = 64
  steps: int = 12
  collision_backend: str = "primitive_boxes"


def main() -> None:
  args = tyro.cli(DebugConfig)
  device = configure_runtime(args.device)
  env, _agent_cfg = build_paired_env(
    args.task,
    args.pair_manifest,
    play=False,
    device=device,
    num_envs=args.num_envs,
    no_terminations=True,
    collision_backend=args.collision_backend,
  )
  try:
    cmd = env.command_manager.get_term("motion")
    assert isinstance(cmd.cfg, MotionCommandCfg)
    cmd.cfg.sampling_mode = "start"
    body_names = list(cmd.cfg.body_names)
    ee_ids = [
      body_names.index("left_ankle_roll_link"),
      body_names.index("right_ankle_roll_link"),
      body_names.index("left_wrist_yaw_link"),
      body_names.index("right_wrist_yaw_link"),
    ]
    env.reset()
    zero = torch.zeros(env.action_space.shape, device=env.device)
    for step in range(args.steps + 1):
      err = torch.abs(
        cmd.body_pos_relative_w[:, ee_ids, 2] - cmd.robot_body_pos_w[:, ee_ids, 2]
      )
      anchor = torch.abs(cmd.anchor_pos_w[:, 2] - cmd.robot_anchor_pos_w[:, 2])
      print(
        f"state={step:02d} "
        f"max_ee_z={float(err.max()):.6f} "
        f"max_anchor_z={float(anchor.max()):.6f}"
      )
      if step < args.steps:
        env.step(zero)
  finally:
    env.close()


if __name__ == "__main__":
  main()
```

- [ ] **Step 2: Add shell wrapper**

Create `scripts/debug_zero_action_collision_backend.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

PAIR_MANIFEST="${1:-/tmp/tt_converted/platform_001/pair.json}"
BACKEND="${2:-primitive_boxes}"

uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.debug_zero_action \
  --pair-manifest "${PAIR_MANIFEST}" \
  --collision-backend "${BACKEND}" \
  --num-envs 64 \
  --steps 12
```

- [ ] **Step 3: Run diagnostic on platform_001**

Run:

```bash
bash scripts/debug_zero_action_collision_backend.sh /tmp/tt_converted/platform_001/pair.json primitive_boxes
```

Expected: primitive box backend should track closer to plane/box behavior than hfield, and should not show the previous hfield-only step-7 early failure.

---

### Task 6: Update Terrain Collision Contract

**Files:**
- Modify: `docs/terrain-collision-contract.md`
- Modify: `docs/2026-04-28-terrain-collision-and-scaling-progress.md`

- [ ] **Step 1: Document backend distinction**

Add this section to `docs/terrain-collision-contract.md`:

```markdown
## Runtime Collision Backends

`terrain_hf.npy` is source terrain data, not a mandate to use MuJoCo `hfield` collision.

terrain_tracking supports these runtime backends:

- `primitive_boxes`: default for PARC platform/block terrain. Converts piecewise-constant hf regions into merged MuJoCo box geoms and places them in a shared terrain tile.
- `hfield`: debug backend. Uses MuJoCo hfield collision directly.
- `mesh`: legacy compatibility backend. Uses OBJ mesh collision and is not recommended for non-convex terrain.

For large training, terrain geoms should scale with unique terrain tiles, not with `num_envs`.
```

- [ ] **Step 2: Run docs grep sanity check**

Run:

```bash
rg -n "primitive_boxes|hfield|mesh|unique terrain tiles" docs
```

Expected: output includes the new backend section and progress summary.

---

## Final Verification

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest pytest -q tests/test_primitive_box_terrain.py tests/test_env_smoke.py tests/test_cli_help.py
```

Expected: PASS.

Then run the diagnostic:

```bash
bash scripts/debug_zero_action_collision_backend.sh /tmp/tt_converted/platform_001/pair.json primitive_boxes
```

Expected: primitive box backend does not reproduce the previous hfield-specific step-7 zero-action pathology.

## Self-Review

- Spec coverage: This plan addresses both collision backend correctness and geom/env scaling.
- Placeholder scan: No unresolved placeholder markers or unspecified test work remain.
- Type consistency: `collision_backend`, `PairPrimitiveBoxTerrainCfg`, and `make_primitive_box_tile_spec_fn` are used consistently across tasks.
