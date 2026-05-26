# Terrain Collision Status

This note tracks unstable terrain-collision work. Stable field definitions and
backend names live in `terrain-collision-contract.md`. Legacy mesh transform
parameters live in `pair-manifest-terrain-params.md`.

## Decisions

- **OBJ mesh collision is invalid for PARC multi-box terrain.** MuJoCo mesh
  geoms use convex-hull-style collision; concavities and gaps can be filled or
  bridged. Keep `multi_boxes.obj` for upstream retargeting and visualization
  only.
- **`hfield` is a debug backend, not the default training path.** It avoids
  convex-hull filling but MuJoCo-Warp hfield contact changed early foot support
  enough to cause pathological tracking failure under the current G1 setup,
  including on flat all-zero heightfields. Do not retry hfield as the default
  PARC training backend.
- **`primitive_boxes` is the default backend** when `terrain_collision_file` is
  present. It converts piecewise-constant PARC heightfields into merged MuJoCo
  box geoms.
- **`mesh` remains a legacy fallback** for pair manifests without collision
  manifests or explicit compatibility checks.
- **`omniretarget_boxes` is a separate backend** for OmniRetarget URDF terrain;
  it does not consume `terrain_collision.json`.

## Implemented

- Pair runtime reads `terrain_collision.json` plus `terrain_hf.npy` when present.
- Default path: `apply_pair_manifest_to_env_cfg(..., collision_backend="primitive_boxes")`
  routes through mjlab `TerrainGeneratorCfg` with one tile (`num_rows=1`,
  `num_cols=1`) and `env_spacing=0.0`, so terrain geoms do not scale with
  `num_envs` on the primitive-box path.
- Explicit backends: `hfield`, `mesh`, and `omniretarget_boxes` remain selectable
  from train/play CLI flags.
- Converters: `terrain_tracking.convert_pair` and
  `terrain_tracking.convert_omniretarget_robot_terrain`.

Key code:

- `src/terrain_tracking/runtime/apply_pair.py`
- `src/terrain_tracking/scene/primitive_box_terrain.py`
- `src/terrain_tracking/scene/heightfield_spec.py`
- `src/terrain_tracking/scene/paired_mesh_spec.py`
- `src/terrain_tracking/scene/omniretarget_boxes.py`

## Still Open

- **Multi-pair / multi-tile assignment.** Current single-pair training uses one
  shared tile origin. Future multi-motion training needs unique terrain tiles
  mapped from pair metadata, not one terrain copy per env.
- **Per-env duplication on non-default backends.** `hfield`, `mesh`, and
  `omniretarget_boxes` still inject terrain geoms per env through `spec_fn`.
- **Coordinate conventions.** Cell-center versus cell-corner mapping must stay
  exact; misalignment is the most likely source of terrain/motion mismatch.
- **Box decomposition limits.** Rectangle merge must not create gaps or shrink
  support surfaces; box count must stay bounded before large `num_envs` training
  on complex heightfields.
- **Primitive boxes scope.** Appropriate for piecewise-constant PARC
  platform/block/stair terrain; not a universal replacement for continuous noisy
  heightfields.

## Do Not Retry

- Training PARC platform terrain through the per-env hfield backend as the
  default collision path.
- Using non-convex OBJ mesh collision for PARC multi-box terrain in training.
