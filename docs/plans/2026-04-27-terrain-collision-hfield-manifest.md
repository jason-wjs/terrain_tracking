# Terrain Collision HField Manifest Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Preserve the existing holosoma PARC retargeting workspace while adding engine-friendly terrain collision outputs that terrain_tracking can import as MuJoCo heightfields instead of non-convex OBJ mesh geoms.

**Architecture:** The upstream holosoma pipeline remains responsible for compiling PARC `.pkl` samples and still emits the legacy workspace files used by the current climbing retargeter. It additionally emits `terrain_hf.npy` and `terrain_collision.json` from the original PARC `terrain_data.hf/min_point/dx`, and records those files in the paired output metadata. The downstream terrain_tracking package extends `pair.json` parsing and scene injection so collision comes from the new heightfield manifest, while `multi_boxes.obj` is optional visual/debug data only.

**Tech Stack:** Python, NumPy, JSON, MuJoCo `MjSpec`, mjlab downstream task registration, pytest, uv.

---

## Context And Constraints

- Upstream repo: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma`
- Downstream repo: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking`
- Keep holosoma's current five-file workspace for retargeting:
  - `human_joints.npy`
  - `multi_boxes.obj`
  - `box_assets.xml`
  - `g1_29dof_w_multi_boxes.xml`
  - `multi_boxes.urdf`
- Add two upstream outputs for downstream training:
  - `terrain_hf.npy`
  - `terrain_collision.json`
- Do not modify PyPI `mjlab`.
- Do not use `multi_boxes.obj` as downstream collision for non-convex terrain.
- Keep legacy `terrain_file` mesh behavior in terrain_tracking only as a compatibility fallback, preferably with validation/warning for non-convex meshes.

## Target Data Contract

Example `terrain_collision.json`:

```json
{
  "schema_version": 1,
  "terrain_name": "platform_001",
  "frame": {
    "convention": "z_up",
    "origin": "motion_world"
  },
  "collision": {
    "type": "heightfield",
    "hf_file": "terrain_hf.npy",
    "min_point": [-1.0, -1.0],
    "dx": 0.4,
    "base_z": -0.32592592592592595,
    "height_scale": 0.8148148148148148,
    "xy_scale": 0.8148148148148148
  },
  "visual": {
    "file": "multi_boxes.obj",
    "role": "visual_only"
  },
  "source": {
    "format": "PARC",
    "fields": ["terrain_data.hf", "terrain_data.min_point", "terrain_data.dx"],
    "scale_source": {
      "data_format": "parc_humanoid",
      "robot_type": "g1",
      "robot_height": 1.32,
      "default_human_height": 1.62,
      "rule": "RobotConfig.ROBOT_HEIGHT / MotionDataConfig.default_human_height"
    }
  }
}
```

Interpretation:

```text
world_x = xy_scale * (min_point[0] + i * dx)
world_y = xy_scale * (min_point[1] + j * dx)
world_z = height_scale * hf[i, j]
```

`base_z` controls the hfield thickness/lower extent. It should be expressed in the same scaled world frame as `world_z`, or the schema must explicitly state otherwise.

Scale contract:

- `terrain_hf.npy` stores the original PARC `terrain_data.hf` values, not a filename-derived or downstream-rescaled copy.
- `min_point` and `dx` store the original PARC `terrain_data.min_point` and `terrain_data.dx` values.
- `xy_scale` and `height_scale` must be written by holosoma from the actual retargeting scale used for that sample.
- For the current PARC G1 default, after changing `parc_humanoid.default_human_height` to `1.62`, the default scale is `1.32 / 1.62 = 0.8148148148148148`. This number is illustrative only; implementation must compute it from config, not hardcode it.
- If `RobotConfig(robot_height=...)`, `MotionDataConfig(default_human_height=...)`, or `default_scale_factor` is overridden, regenerated `terrain_collision.json` must reflect that override.
- Current `parc_process.py` uses `RetargetingConfig(augmentation=False)`, so `xy_scale == height_scale == scale_factor`. If climbing augmentation is later enabled, include object scale explicitly, for example `xy_scale = scale_factor * object_scale_xy` and `height_scale = scale_factor * object_scale_z`.
- Downstream terrain_tracking must treat manifest scale as source of truth. It must not infer scale from OBJ mesh filenames such as `0.74_0.74_0.74`.

---

### Task 1: Add Upstream Terrain Collision Exporter

**Files:**
- Modify: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma/src/holosoma_retargeting/holosoma_retargeting/parc_process/terrain_scene.py`
- Test: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma/tests/test_parc_terrain_collision_export.py`

**Step 1: Write failing exporter tests**

Create `tests/test_parc_terrain_collision_export.py`:

```python
from pathlib import Path
import json

import numpy as np

from holosoma_retargeting.parc_process.source_io import ParcTerrainData
from holosoma_retargeting.parc_process.terrain_scene import export_parc_scene


def test_export_parc_scene_writes_heightfield_collision_manifest(tmp_path: Path) -> None:
    terrain = ParcTerrainData(
        hf=np.array([[0.0, 0.2], [0.4, 0.6]], dtype=np.float32),
        hf_maxmin=np.array([0.6, 0.0], dtype=np.float32),
        min_point=np.array([-1.0, -2.0], dtype=np.float32),
        dx=0.4,
    )

    assets = export_parc_scene(
        terrain,
        tmp_path,
        object_name="multi_boxes",
        scale_factor=0.5,
        scale_source={
            "robot_type": "test_bot",
            "robot_height": 1.0,
            "data_format": "parc_humanoid",
            "default_human_height": 2.0,
            "rule": "test scale",
        },
    )

    assert assets.terrain_hf_path.is_file()
    assert assets.terrain_collision_path.is_file()
    np.testing.assert_allclose(np.load(assets.terrain_hf_path), terrain.hf)

    manifest = json.loads(assets.terrain_collision_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["terrain_name"] == "multi_boxes"
    assert manifest["collision"]["type"] == "heightfield"
    assert manifest["collision"]["hf_file"] == "terrain_hf.npy"
    assert manifest["collision"]["min_point"] == [-1.0, -2.0]
    assert manifest["collision"]["dx"] == 0.4
    assert manifest["collision"]["xy_scale"] == 0.5
    assert manifest["collision"]["height_scale"] == 0.5
    assert manifest["source"]["scale_source"]["default_human_height"] == 2.0
    assert manifest["visual"]["file"] == "multi_boxes.obj"
    assert manifest["visual"]["role"] == "visual_only"
```

**Step 2: Run test to verify it fails**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma
uv run pytest -q tests/test_parc_terrain_collision_export.py
```

Expected: FAIL because `ParcSceneAssets` has no `terrain_hf_path` / `terrain_collision_path`, and no exporter writes those files.

**Step 3: Implement minimal exporter**

In `terrain_scene.py`:

- Add fields to `ParcSceneAssets`:

```python
terrain_hf_path: Path
terrain_collision_path: Path
```

- Add helpers:

```python
import json


def _write_terrain_hf(hf: np.ndarray, output_path: Path) -> Path:
    np.save(output_path, np.asarray(hf, dtype=np.float32))
    return output_path


def _write_terrain_collision_manifest(
    *,
    output_path: Path,
    object_name: str,
    hf_path: Path,
    obj_path: Path,
    terrain_data: ParcTerrainData,
    base_z: float,
    scale_factor: float,
    scale_source: dict[str, float | str] | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "terrain_name": object_name,
        "frame": {"convention": "z_up", "origin": "motion_world"},
        "collision": {
            "type": "heightfield",
            "hf_file": hf_path.name,
            "min_point": [float(terrain_data.min_point[0]), float(terrain_data.min_point[1])],
            "dx": float(terrain_data.dx),
            "base_z": float(base_z * scale_factor),
            "xy_scale": float(scale_factor),
            "height_scale": float(scale_factor),
        },
        "visual": {"file": obj_path.name, "role": "visual_only"},
        "source": {
            "format": "PARC",
            "fields": ["terrain_data.hf", "terrain_data.min_point", "terrain_data.dx"],
            "scale_source": scale_source or {},
        },
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
```

- In `export_parc_scene()`, compute `base_z` once and pass it to both OBJ and manifest generation:

```python
base_z = float(np.min(terrain_data.hf)) - max(float(terrain_data.dx), 0.1)
_write_obj(..., base_z=base_z)  # adjust _write_obj signature
terrain_hf_path = out_dir / "terrain_hf.npy"
terrain_collision_path = out_dir / "terrain_collision.json"
_write_terrain_hf(terrain_data.hf, terrain_hf_path)
_write_terrain_collision_manifest(...)
```

Change `export_parc_scene()` to accept:

```python
scale_factor: float = 1.0
scale_source: dict[str, float | str] | None = None
```

Do not compute the retarget scale inside `terrain_scene.py`; this module should only receive and serialize the scale chosen by the PARC process entrypoint.

**Step 4: Run test to verify it passes**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma
uv run pytest -q tests/test_parc_terrain_collision_export.py
```

Expected: PASS.

**Step 5: Commit**

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma
git add src/holosoma_retargeting/holosoma_retargeting/parc_process/terrain_scene.py tests/test_parc_terrain_collision_export.py
git commit -m "feat: export PARC heightfield collision manifest"
```

---

### Task 2: Thread Collision Assets Through Upstream Workspace And Paired Output

**Files:**
- Modify: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma/src/holosoma_retargeting/holosoma_retargeting/parc_process/workspace.py`
- Modify: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma/src/holosoma_retargeting/holosoma_retargeting/parc_process/output_writer.py`
- Test: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma/tests/test_parc_process_output.py`

**Step 1: Write failing tests**

Add or extend an upstream test:

```python
def test_parc_workspace_exposes_collision_assets(tmp_path: Path, sample: ParcSample, source_xml: Path) -> None:
    workspace = build_parc_workspace(
        sample=sample,
        source_xml=source_xml,
        output_dir=tmp_path,
        task_name="demo",
        scale_factor=0.5,
        scale_source={"rule": "test scale"},
    )

    assert workspace.terrain_hf_path.is_file()
    assert workspace.terrain_collision_path.is_file()
```

Add paired output metadata assertion:

```python
def test_paired_output_records_collision_assets(tmp_path: Path, sample: ParcSample) -> None:
    result = write_paired_output(
        qpos=make_qpos(),
        source_sample=sample,
        output_root=tmp_path,
        motion_name="demo_g1",
        scale_factor=0.5,
        workspace_path=sample.misc_data["workspace_path"],
        terrain_collision_path=sample.misc_data["terrain_collision_path"],
        terrain_hf_path=sample.misc_data["terrain_hf_path"],
    )

    loaded = result.load_motion_file()
    assert loaded.misc_data["parc_process:terrain_collision_file"].endswith("terrain_collision.json")
    assert loaded.misc_data["parc_process:terrain_hf_file"].endswith("terrain_hf.npy")
```

Adapt fixture names to existing upstream tests. Keep the test focused on metadata, not full retargeting.

**Step 2: Run tests to verify failure**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma
uv run pytest -q tests/test_parc_process_output.py tests/test_parc_terrain_collision_export.py
```

Expected: FAIL because workspace/result metadata does not expose collision assets yet.

**Step 3: Implement workspace fields**

In `workspace.py`, extend `ParcWorkspace`:

```python
terrain_hf_path: Path
terrain_collision_path: Path
```

Return the fields from `build_parc_workspace()`:

```python
terrain_hf_path=scene.terrain_hf_path,
terrain_collision_path=scene.terrain_collision_path,
```

Also extend `build_parc_workspace()` with:

```python
scale_factor: float = 1.0
scale_source: dict[str, float | str] | None = None
```

and pass both through to `export_parc_scene()`. This keeps workspace generation aligned with the retarget scale without making `terrain_scene.py` depend on retargeting configs.

**Step 4: Implement paired output metadata**

In `output_writer.py`, extend `write_paired_output()` parameters:

```python
terrain_collision_path: str | Path | None = None
terrain_hf_path: str | Path | None = None
terrain_visual_path: str | Path | None = None
```

Extend `_misc_payload()` to write:

```python
if terrain_collision_path is not None:
    misc_data["parc_process:terrain_collision_file"] = str(Path(terrain_collision_path).expanduser().resolve())
if terrain_hf_path is not None:
    misc_data["parc_process:terrain_hf_file"] = str(Path(terrain_hf_path).expanduser().resolve())
if terrain_visual_path is not None:
    misc_data["parc_process:terrain_visual_file"] = str(Path(terrain_visual_path).expanduser().resolve())
```

In `examples/parc_process.py`, compute scale before building the workspace:

```python
scale_factor, scale_source = _default_scale_factor_with_source(robot_type)
```

Use the actual config values:

```python
def _default_scale_factor_with_source(
    robot_type: str,
    data_format: str = "parc_humanoid",
) -> tuple[float, dict[str, float | str]]:
    robot_cfg = RobotConfig(robot_type=robot_type)
    motion_cfg = MotionDataConfig(data_format=data_format, robot_type=robot_type)
    if motion_cfg.default_scale_factor is not None:
        return float(motion_cfg.default_scale_factor), {
            "robot_type": robot_type,
            "data_format": data_format,
            "default_scale_factor": float(motion_cfg.default_scale_factor),
            "rule": "MotionDataConfig.default_scale_factor",
        }
    if motion_cfg.default_human_height is not None:
        return float(robot_cfg.ROBOT_HEIGHT / motion_cfg.default_human_height), {
            "robot_type": robot_type,
            "robot_height": float(robot_cfg.ROBOT_HEIGHT),
            "data_format": data_format,
            "default_human_height": float(motion_cfg.default_human_height),
            "rule": "RobotConfig.ROBOT_HEIGHT / MotionDataConfig.default_human_height",
        }
    return 1.0, {
        "robot_type": robot_type,
        "data_format": data_format,
        "rule": "identity fallback",
    }
```

Pass `scale_factor` and `scale_source` to `build_parc_workspace()` before retargeting runs, and reuse the same `scale_factor` for `write_paired_output()`. Do not call `_default_scale_factor()` twice with potentially different config state.

In `examples/parc_process.py`, pass collision assets:

```python
terrain_collision_path=workspace.terrain_collision_path,
terrain_hf_path=workspace.terrain_hf_path,
terrain_visual_path=workspace.obj_path,
```

**Step 5: Run tests to verify pass**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma
uv run pytest -q tests/test_parc_terrain_collision_export.py tests/test_parc_process_output.py
```

Expected: PASS.

**Step 6: Commit**

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma
git add src/holosoma_retargeting/holosoma_retargeting/parc_process/workspace.py \
  src/holosoma_retargeting/holosoma_retargeting/parc_process/output_writer.py \
  src/holosoma_retargeting/holosoma_retargeting/examples/parc_process.py \
  tests/test_parc_process_output.py
git commit -m "feat: thread terrain collision assets through PARC outputs"
```

---

### Task 3: Add Downstream Collision Manifest Parser

**Files:**
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/runtime/terrain_collision.py`
- Modify: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/runtime/pair_manifest.py`
- Test: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/tests/test_terrain_collision_manifest.py`
- Test: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/tests/test_pair_manifest.py`

**Step 1: Write failing tests**

Create `tests/test_terrain_collision_manifest.py`:

```python
from pathlib import Path
import json

import numpy as np

from terrain_tracking.runtime.terrain_collision import TerrainCollisionManifest


def test_load_heightfield_collision_manifest_resolves_relative_hf(tmp_path: Path) -> None:
    np.save(tmp_path / "terrain_hf.npy", np.array([[0.0, 0.2], [0.4, 0.6]], dtype=np.float32))
    manifest_path = tmp_path / "terrain_collision.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "terrain_name": "demo",
                "frame": {"convention": "z_up", "origin": "motion_world"},
                "collision": {
                    "type": "heightfield",
                    "hf_file": "terrain_hf.npy",
                    "min_point": [-1.0, -2.0],
                    "dx": 0.4,
                    "base_z": -0.4,
                    "xy_scale": 0.5,
                    "height_scale": 0.75,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = TerrainCollisionManifest.load(manifest_path)

    assert manifest.type == "heightfield"
    assert manifest.hf_file == (tmp_path / "terrain_hf.npy").resolve()
    assert manifest.min_point == (-1.0, -2.0)
    assert manifest.dx == 0.4
    assert manifest.base_z == -0.4
    assert manifest.xy_scale == 0.5
    assert manifest.height_scale == 0.75
```

Extend `tests/test_pair_manifest.py`:

```python
def test_pair_manifest_load_reads_collision_manifest_path(tmp_path: Path) -> None:
    motion = create_motion_clip(tmp_path / "motion.npz")
    collision = tmp_path / "terrain_collision.json"
    collision.write_text("{}", encoding="utf-8")
    terrain = create_quad_obj(tmp_path / "terrain.obj")
    pair_path = create_pair_manifest(
        tmp_path / "pair.json",
        motion_file=motion.name,
        terrain_file=terrain.name,
    )
    payload = json.loads(pair_path.read_text(encoding="utf-8"))
    payload["terrain_collision_file"] = collision.name
    pair_path.write_text(json.dumps(payload), encoding="utf-8")

    pair = PairManifest.load(pair_path)

    assert pair.terrain_collision_file == collision.resolve()
```

**Step 2: Run tests to verify failure**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
uv run pytest -q tests/test_terrain_collision_manifest.py tests/test_pair_manifest.py
```

Expected: FAIL because parser/file fields do not exist.

**Step 3: Implement parser**

Create `runtime/terrain_collision.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


def _required_float_pair(payload: dict[str, Any], key: str) -> tuple[float, float]:
    value = payload.get(key)
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"collision.{key} must contain 2 numbers")
    return (float(value[0]), float(value[1]))


@dataclass(frozen=True)
class TerrainCollisionManifest:
    type: Literal["heightfield"]
    hf_file: Path
    min_point: tuple[float, float]
    dx: float
    base_z: float
    xy_scale: float = 1.0
    height_scale: float = 1.0

    @classmethod
    def load(cls, path: str | Path) -> "TerrainCollisionManifest":
        manifest_path = Path(path).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Terrain collision manifest must be a JSON object")
        collision = payload.get("collision")
        if not isinstance(collision, dict):
            raise ValueError("Terrain collision manifest is missing collision object")
        if collision.get("type") != "heightfield":
            raise ValueError(f"Unsupported terrain collision type: {collision.get('type')!r}")
        hf_value = collision.get("hf_file")
        if not isinstance(hf_value, str) or not hf_value:
            raise ValueError("collision.hf_file must be a non-empty string")
        hf_file = (manifest_path.parent / hf_value).resolve()
        if not hf_file.exists():
            raise FileNotFoundError(hf_file)
        return cls(
            type="heightfield",
            hf_file=hf_file,
            min_point=_required_float_pair(collision, "min_point"),
            dx=float(collision["dx"]),
            base_z=float(collision["base_z"]),
            xy_scale=float(collision.get("xy_scale", 1.0)),
            height_scale=float(collision.get("height_scale", 1.0)),
        )
```

**Step 4: Extend PairManifest**

In `pair_manifest.py`:

- Add optional field:

```python
terrain_collision_file: Path | None = None
terrain_visual_file: Path | None = None
```

- Add helper `_optional_path()`.
- Load `terrain_collision_file` and `terrain_visual_file` if present.
- Keep `terrain_file` required for one compatibility pass, or relax it only after conversion scripts are updated.

**Step 5: Run tests to verify pass**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
uv run pytest -q tests/test_terrain_collision_manifest.py tests/test_pair_manifest.py
```

Expected: PASS.

**Step 6: Commit**

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
git add src/terrain_tracking/runtime/terrain_collision.py \
  src/terrain_tracking/runtime/pair_manifest.py \
  tests/test_terrain_collision_manifest.py \
  tests/test_pair_manifest.py
git commit -m "feat: parse terrain collision manifests"
```

---

### Task 4: Add Downstream MuJoCo Heightfield Terrain Injection

**Files:**
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/scene/heightfield_spec.py`
- Modify: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/runtime/apply_pair.py`
- Test: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/tests/test_heightfield_spec.py`
- Test: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/tests/test_env_smoke.py`

**Step 1: Write failing heightfield spec test**

Create `tests/test_heightfield_spec.py`:

```python
from pathlib import Path
import json

import mujoco
import numpy as np

from terrain_tracking.runtime.terrain_collision import TerrainCollisionManifest
from terrain_tracking.scene.heightfield_spec import make_heightfield_spec_fn


def _write_collision(tmp_path: Path) -> TerrainCollisionManifest:
    np.save(tmp_path / "terrain_hf.npy", np.array([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32))
    path = tmp_path / "terrain_collision.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "terrain_name": "demo",
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
    return TerrainCollisionManifest.load(path)


def test_make_heightfield_spec_fn_adds_hfield_geom_per_env(tmp_path: Path) -> None:
    manifest = _write_collision(tmp_path)
    spec = mujoco.MjSpec()
    spec_fn = make_heightfield_spec_fn(manifest, num_envs=3, env_spacing=2.0)

    spec_fn(spec)

    assert len(spec.hfields) == 1
    paired_bodies = [body for body in spec.worldbody.bodies if body.name.startswith("paired_terrain_")]
    assert len(paired_bodies) == 3
    assert all(body.geoms[0].type == mujoco.mjtGeom.mjGEOM_HFIELD for body in paired_bodies)
    assert paired_bodies[0].pos[0] == 0.0
    assert paired_bodies[0].pos[1] == 0.0
```

If MuJoCo Python `MjSpec` uses different hfield property names, inspect the local MuJoCo API and adapt the test to the actual fields. Keep the test intent unchanged.

**Step 2: Run test to verify failure**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
uv run pytest -q tests/test_heightfield_spec.py
```

Expected: FAIL because `heightfield_spec.py` does not exist.

**Step 3: Implement heightfield spec helper**

Create `scene/heightfield_spec.py`.

Implementation requirements:

- Load `terrain_hf.npy`.
- Apply `height_scale` to height values.
- Compute hfield physical size from shape, `dx`, and `xy_scale`.
- Compute the hfield center from `min_point`, `dx`, shape, and `xy_scale`; do not drop the original terrain xy origin.
- Add one MuJoCo hfield asset.
- Add one hfield geom per env origin.
- Set `contype=1`, `conaffinity=1`.
- Do not add visual OBJ collision.

Pseudo-code:

```python
from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np

from terrain_tracking.runtime.terrain_collision import TerrainCollisionManifest
from terrain_tracking.scene.paired_mesh_spec import compute_env_origins_grid


def _load_heightfield(manifest: TerrainCollisionManifest) -> np.ndarray:
    hf = np.load(manifest.hf_file).astype(np.float32)
    if hf.ndim != 2:
        raise ValueError(f"heightfield must be 2D, got shape {hf.shape}")
    return hf * np.float32(manifest.height_scale)


def make_heightfield_spec_fn(
    manifest: TerrainCollisionManifest,
    *,
    num_envs: int,
    env_spacing: float,
    hfield_name: str = "paired_terrain_hfield",
) -> Callable[[mujoco.MjSpec], None]:
    hf = _load_heightfield(manifest)
    env_origins = compute_env_origins_grid(num_envs=num_envs, env_spacing=env_spacing)
    nx, ny = hf.shape
    size_x = float((nx - 1) * manifest.dx * manifest.xy_scale)
    size_y = float((ny - 1) * manifest.dx * manifest.xy_scale)
    center_x = float(manifest.xy_scale * (manifest.min_point[0] + 0.5 * (nx - 1) * manifest.dx))
    center_y = float(manifest.xy_scale * (manifest.min_point[1] + 0.5 * (ny - 1) * manifest.dx))
    z_min = min(float(manifest.base_z), float(hf.min()))
    z_max = float(hf.max())

    def spec_fn(spec: mujoco.MjSpec) -> None:
        # Adapt this block to actual MuJoCo MjSpec API:
        hfield = spec.add_hfield(name=hfield_name, nrow=nx, ncol=ny)
        hfield.size = [size_x / 2, size_y / 2, max(z_max - z_min, 1e-6), abs(z_min)]
        hfield.data = ((hf - z_min) / max(z_max - z_min, 1e-6)).reshape(-1).tolist()
        for env_id, env_origin in enumerate(env_origins):
            body_pos = env_origin.astype(np.float64).copy()
            body_pos[0] += center_x
            body_pos[1] += center_y
            body = spec.worldbody.add_body(name=f"paired_terrain_{env_id}", pos=body_pos.tolist())
            body.add_geom(
                name=f"paired_terrain_{env_id}",
                type=mujoco.mjtGeom.mjGEOM_HFIELD,
                hfieldname=hfield_name,
                contype=1,
                conaffinity=1,
            )

    return spec_fn
```

Before finalizing this implementation, inspect the local `mujoco.MjSpec` hfield API using a short REPL command. Do not guess property names.

**Step 4: Route apply_pair through heightfield when available**

In `apply_pair.py`:

```python
if pair.terrain_collision_file is not None:
    collision = TerrainCollisionManifest.load(pair.terrain_collision_file)
    make_heightfield_spec_fn(collision, num_envs=cfg.scene.num_envs, env_spacing=cfg.scene.env_spacing)(spec)
else:
    make_paired_mesh_spec_fn(...)
```

**Step 5: Add env smoke test**

Extend `tests/test_env_smoke.py`:

- Build a tiny pair with `terrain_collision_file`.
- Instantiate env on CPU with `num_envs=1`.
- Assert compiled model has a `MJGEOM_HFIELD` paired terrain geom.
- Assert no `paired_terrain_0` mesh geom is used for collision in this path.

**Step 6: Run tests**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
uv run pytest -q tests/test_heightfield_spec.py tests/test_env_smoke.py
```

Expected: PASS.

**Step 7: Commit**

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
git add src/terrain_tracking/scene/heightfield_spec.py \
  src/terrain_tracking/runtime/apply_pair.py \
  tests/test_heightfield_spec.py \
  tests/test_env_smoke.py
git commit -m "feat: inject paired terrain as MuJoCo heightfield"
```

---

### Task 5: Update Downstream Pair Conversion To Prefer Collision Manifests

**Files:**
- Modify: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/src/terrain_tracking/convert_pair.py`
- Modify: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/scripts/convert_pair.sh`
- Test: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/tests/test_data_conversion_convert_pair.py`
- Test: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/tests/test_cli_help.py`

**Step 1: Write failing tests**

Extend conversion test:

```python
def test_convert_pair_writes_collision_manifest_reference(tmp_path: Path) -> None:
    motion = create_motion_clip(tmp_path / "motion.npz")
    terrain = create_ramp_obj(tmp_path / "terrain.obj")
    collision = tmp_path / "terrain_collision.json"
    collision.write_text("{}", encoding="utf-8")

    bundle_dir = convert_pair(
        ConvertPairConfig(
            motion_file=motion,
            terrain_file=terrain,
            terrain_collision_file=collision,
            output_dir=tmp_path / "converted",
            sample_name="demo",
        )
    )

    payload = json.loads((bundle_dir / "pair.json").read_text(encoding="utf-8"))
    assert payload["terrain_collision_file"] == str(collision.resolve())
    assert payload["terrain_file"] == str(terrain.resolve())
```

Extend CLI help test:

```python
assert "--terrain-collision-file" in proc.stdout
assert "--terrain-visual-file" in proc.stdout
```

**Step 2: Run tests to verify failure**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
uv run pytest -q tests/test_data_conversion_convert_pair.py tests/test_cli_help.py
```

Expected: FAIL because conversion config and CLI do not support collision files.

**Step 3: Implement conversion fields**

In `ConvertPairConfig`, add:

```python
terrain_collision_file: str | Path | None = None
terrain_visual_file: str | Path | None = None
```

In `write_pair_manifest_bundle()`, write optional references:

```python
if terrain_collision_file is not None:
    payload["terrain_collision_file"] = _validate_manifest_reference(...)
if terrain_visual_file is not None:
    payload["terrain_visual_file"] = _validate_manifest_reference(...)
```

Keep `terrain_file` for compatibility during this migration.

**Step 4: Add CLI flags**

In `build_arg_parser()`:

```python
parser.add_argument("--terrain-collision-file", default=None)
parser.add_argument("--terrain-visual-file", default=None)
```

Thread these into `ConvertPairConfig`.

**Step 5: Update shell wrapper**

In `scripts/convert_pair.sh`, pass:

```bash
--terrain-collision-file /tmp/parc_process_workspace/workspace/platform_001/terrain_collision.json \
--terrain-visual-file /tmp/parc_process_workspace/workspace/platform_001/multi_boxes.obj \
```

Keep `--terrain-file` temporarily equal to the visual OBJ until all downstream callers move to `terrain_collision_file`.

**Step 6: Run tests**

Run:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
uv run pytest -q tests/test_data_conversion_convert_pair.py tests/test_cli_help.py tests/test_pair_manifest.py
```

Expected: PASS.

**Step 7: Commit**

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
git add src/terrain_tracking/convert_pair.py scripts/convert_pair.sh \
  tests/test_data_conversion_convert_pair.py tests/test_cli_help.py
git commit -m "feat: include terrain collision manifests in pair bundles"
```

---

### Task 6: Add Cross-Repo Smoke Verification For Real PARC Sample

**Files:**
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/scripts/debug_validate_hfield_pair.sh`
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/tests/test_real_pair_contact_smoke.py` if stable fixture paths are available; otherwise keep as a script-only manual verification.

**Step 1: Add validation script**

Create `scripts/debug_validate_hfield_pair.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

PAIR_MANIFEST="${1:?usage: $0 /path/to/pair.json}"

uv run python - <<'PY' "${PAIR_MANIFEST}"
import sys
import mujoco
import torch

from terrain_tracking.tasks.blind_terrain_tracking.scripts.common import build_paired_env

pair_manifest = sys.argv[1]
env, _ = build_paired_env(
    "TT-Tracking-TerrainBlind-Unitree-G1",
    pair_manifest,
    play=True,
    device="cpu",
    num_envs=1,
    no_terminations=True,
)
try:
    model = env.unwrapped.sim.mj_model
    data = env.unwrapped.sim.mj_data
    env.reset()
    terrain_geoms = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if name.startswith("paired_terrain_"):
            terrain_geoms.append((name, int(model.geom_type[gid])))
    print("terrain_geoms", terrain_geoms)
    assert any(gtype == mujoco.mjtGeom.mjGEOM_HFIELD for _, gtype in terrain_geoms)
    action = torch.zeros((1, env.unwrapped.single_action_space.shape[0]), device=env.device)
    for _ in range(5):
        env.step(action)
    max_penetration = 0.0
    for cid in range(data.ncon):
        c = data.contact[cid]
        max_penetration = min(max_penetration, float(c.dist))
    print("ncon", data.ncon, "max_penetration", max_penetration)
finally:
    env.close()
PY
```

**Step 2: Run full sample pipeline manually**

Run upstream:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma
uv run python -m holosoma_retargeting.examples.parc_process \
  --sample /abs/path/to/platform_001.pkl \
  --source-xml /abs/path/to/humanoid.xml \
  --output-root /tmp/parc_g1_outputs \
  --retarget-save-dir /tmp/parc_process_workspace
```

Expected:

```text
/tmp/parc_process_workspace/workspace/platform_001/terrain_hf.npy
/tmp/parc_process_workspace/workspace/platform_001/terrain_collision.json
```

Run downstream conversion:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
uv run python -m terrain_tracking.convert_pair \
  --motion-file /tmp/tt_converted/platform_001/motion.npz \
  --terrain-file /tmp/parc_process_workspace/workspace/platform_001/multi_boxes.obj \
  --terrain-collision-file /tmp/parc_process_workspace/workspace/platform_001/terrain_collision.json \
  --terrain-visual-file /tmp/parc_process_workspace/workspace/platform_001/multi_boxes.obj \
  --output-dir /tmp/tt_converted \
  --sample-name platform_001
```

The hfield collision scale must come from `terrain_collision.json`. Use `--terrain-scale` only for legacy visual OBJ compatibility if an existing conversion path still requires it; never derive collision scale from the OBJ filename.

Run validation:

```bash
bash scripts/debug_validate_hfield_pair.sh /tmp/tt_converted/platform_001/pair.json
```

Expected:

- paired terrain geom type is `MJGEOM_HFIELD`
- no `paired_terrain_0` collision mesh geom is used
- initial penetration is not in the old `-0.05m` to `-0.08m` range

**Step 3: Commit**

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
git add scripts/debug_validate_hfield_pair.sh
git commit -m "test: add hfield pair validation script"
```

---

### Task 7: Documentation And Migration Notes

**Files:**
- Create: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/docs/terrain-collision-contract.md`
- Modify: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking/README.md`
- Optionally create upstream doc: `/home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma/docs/parc-terrain-collision-export.md`

**Step 1: Write docs**

Document:

- Why non-convex OBJ mesh collision is invalid for this terrain.
- Old holosoma workspace files remain retargeting assets.
- New collision files are downstream training assets.
- `multi_boxes.obj` is visual/debug only for terrain_tracking.
- Heightfield is the default collision representation for PARC `hf`.
- Box primitive collision is deferred to a later phase for high-fidelity small-env experiments.

**Step 2: Update README quick start**

Add new conversion example:

```bash
uv run python -m terrain_tracking.convert_pair \
  --motion-file /tmp/tt_converted/platform_001/motion.npz \
  --terrain-file /tmp/parc_process_workspace/workspace/platform_001/multi_boxes.obj \
  --terrain-collision-file /tmp/parc_process_workspace/workspace/platform_001/terrain_collision.json \
  --terrain-visual-file /tmp/parc_process_workspace/workspace/platform_001/multi_boxes.obj \
  --output-dir /tmp/tt_converted \
  --sample-name platform_001
```

**Step 3: Commit**

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
git add docs/terrain-collision-contract.md README.md
git commit -m "docs: describe terrain collision manifest workflow"
```

---

## Final Verification

Run upstream tests:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/RETARGET/holosoma
uv run pytest -q tests/test_parc_terrain_collision_export.py tests/test_parc_process_output.py
```

Run downstream tests:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
uv run pytest -q tests
```

Run real sample smoke:

```bash
cd /home/humanoid/Projects/Junsong_WU/learning/locomotion/terrain_tracking
bash scripts/debug_validate_hfield_pair.sh /tmp/tt_converted/platform_001/pair.json
```

Expected final state:

- holosoma still emits the legacy retargeting workspace.
- holosoma additionally emits `terrain_hf.npy` and `terrain_collision.json`.
- terrain_tracking pair bundles can reference `terrain_collision_file`.
- terrain_tracking uses MuJoCo hfield collision when `terrain_collision_file` exists.
- non-convex OBJ mesh is no longer the collision source for WBT terrain training.
- reset contact smoke no longer shows immediate `-5cm` to `-8cm` terrain penetration caused by mesh convex hull collision.

## Deferred Work

- Primitive box/column collision manifest for exact step-side collision.
- Visual-only OBJ injection in terrain_tracking with `contype=0`, `conaffinity=0`.
- Migration to make `terrain_file` optional once all scripts use `terrain_collision_file`.
- Retargeter-side conversion from mesh scene to hfield/primitive collision if object non-penetration is re-enabled and must match downstream training exactly.
