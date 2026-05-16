# CoACD Collision Backend Design

## Goal

Add a single-pair `collision_backend="coacd"` option so the current terrain
tracking tasks can compare CoACD mesh collision against the existing
`primitive_boxes` backend without changing the default training path.

This change does not introduce motion-terrain dataset training, active shards,
or InstinctMJ's motion-matched terrain generator.

## Context

The current single-pair runtime is centered on
`apply_pair_manifest_to_env_cfg()`. It accepts one `PairManifest`, sets the
motion command to the pair motion file, and selects one terrain collision
backend:

- `primitive_boxes`: default backend based on `terrain_collision.json` and
  `hf.npy`.
- `hfield`: debug backend based on MuJoCo hfields.
- `mesh`: legacy visual mesh collision.

The new backend should fit this same switch and preserve all existing behavior
unless the user explicitly passes `--collision-backend coacd`.

## Architecture

Create a focused scene module:

```text
src/terrain_tracking/scene/coacd_mesh_spec.py
```

The module loads `pair.terrain_file`, applies the same
`terrain_scale`, `terrain_quat_xyzw`, and `terrain_translation` semantics as
`paired_mesh_spec.py`, runs CoACD approximate convex decomposition, and injects
the resulting convex hulls into `mujoco.MjSpec`.

The source visual mesh and CoACD collision hulls are separate:

- The source terrain mesh is registered for visualization only and has
  `contype=0`, `conaffinity=0`.
- Each CoACD hull is registered as a separate MuJoCo mesh asset and added as a
  collision geom with `contype=1`, `conaffinity=1`, `margin=0.0`, and `gap=0.0`.

This follows the useful part of InstinctMJ's CoACD strategy while keeping the
current project's single-pair manifest and environment setup.

## Dependency

Add `coacd` to the project dependencies. If the import is unavailable at
runtime, `collision_backend="coacd"` should raise a clear error explaining that
the backend requires the `coacd` Python package.

## Cache

CoACD decomposition is cached in memory and on disk. The disk cache lives under
the project work directory rather than next to the dataset mesh:

```text
.cache/terrain_tracking/coacd/
```

The cache key includes:

- cache format version
- terrain file absolute path
- terrain file size and mtime
- terrain scale, quaternion, and translation
- CoACD parameter values

The cache stores one compressed npz containing `num_parts` plus per-part
`verts_N` and `faces_N` arrays.

## CoACD Options

Use a small dataclass for backend options. The first implementation keeps these
as code defaults rather than CLI flags:

- `threshold=0.05`
- `max_convex_hull=-1`
- `preprocess_mode="auto"`
- `preprocess_resolution=50`
- `resolution=2000`
- `mcts_nodes=20`
- `mcts_iterations=150`
- `mcts_max_depth=3`
- `seed=0`
- `pca=False`
- `merge=False`
- `decimate=False`
- `max_ch_vertex=256`
- `extrude=False`
- `extrude_margin=0.01`
- `apx_mode="ch"`
- `geom_margin=0.0`
- `z_offset=0.0`
- `visualize_collision_hulls=False`

The values mirror InstinctMJ's conservative terrain defaults. CLI exposure is
deferred until single-pair experiments show which parameters are worth tuning.

## Error Handling

The backend should fail early with actionable messages when:

- `coacd` is not installed.
- CoACD produces no valid hulls.
- A hull has degenerate vertices/faces that would make MuJoCo compilation
  unreliable.

Degenerate hulls are skipped when possible. If all hulls are skipped, the
backend raises `ValueError`.

## Testing

Add focused unit tests that cover:

- `apply_pair_manifest_to_env_cfg()` accepts `collision_backend="coacd"`.
- A simple mesh pair produces source visual mesh geoms and collision hull geoms.
- Source visual mesh collision is disabled.
- CoACD hull geoms have collision enabled, zero margin, and zero gap.
- Disk cache can be loaded without re-running decomposition.
- Missing `coacd` import raises a clear backend-specific error.

Baseline note: before this work, the full test suite in the isolated worktree
has one unrelated failure:

```text
tests/test_cli_help.py::test_train_shell_uses_mjlab_env_spacing_flag
```

The CoACD work will be verified with focused tests and the full suite will be
reported with this pre-existing failure if it remains.

## Non-Goals

- Do not change `primitive_boxes` behavior.
- Do not make CoACD the default backend.
- Do not add dataset or active-shard training support.
- Do not copy InstinctMJ's `MotionMatchedTerrainCfg`, metadata format, crop
  convention, or prewarm-all dataset behavior.
