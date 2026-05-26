# Design History

Concise record of design rationale for shipped features. New specs use
`YYYY-MM-DD-<slug>-design.md` in this directory.

## Pair Manifest As Runtime Input

- One `pair.json` connects one motion clip to one terrain source for train/play.
- Converters write self-contained pair bundles; runtime code does not hard-code
  dataset paths.
- `terrain_file` remains for compatibility; PARC training prefers
  `terrain_collision_file` when present.

## Collision Backend Strategy

- PARC multi-box OBJ meshes are unsafe as MuJoCo mesh collision because convex
  hull collision fills concavities.
- `terrain_collision.json` plus `terrain_hf.npy` is the engine-friendly source
  data contract; runtime chooses how to instantiate collision geometry.
- Default PARC path: decompose piecewise-constant heightfields into primitive
  boxes on a shared terrain tile.
- `hfield` and `mesh` remain explicit non-default paths for debug and legacy
  compatibility. See `docs/terrain-collision-status.md` for open work.

## Oracle Perception Tasks

- Add simulation-only oracle tasks on top of blind tracking without changing
  blind behavior, rewards, or terminations.
- `TerrainOracleHeight`: blind observations plus a clean local height scan from
  `torso_link` (`0.7 m x 0.7 m` grid).
- `TerrainOracleTeacher`: OracleHeight plus PHP-style teacher observations.
  Tracking anchor stays `torso_link`; teacher proprioception and privileged
  terms use pelvis frame semantics while height scan remains torso-mounted.
- Out of scope: deployment policies, depth images, DAgger, student distillation,
  observation noise, and long-run convergence claims.

## OmniRetarget Robot-Terrain

- One OmniRetarget clip plus external URDF/box assets converts into a normal
  pair bundle.
- Motion is resampled to 50 Hz and forward-kinematics body states are computed
  from the mjlab G1 model, not the OmniRetarget visualization URDF.
- Collision uses `omniretarget_boxes`, a separate backend from PARC
  `primitive_boxes`, even though both emit MuJoCo box geoms.
- Stage scope: single-pair end-to-end smoke, not a multi-sample dataset sampler.

## Lifecycle

1. Add a dated design spec here before or during implementation.
2. When the design ships, add or update a section in this file.
3. Delete or archive the detailed spec; git retains the full history.
4. Promote stable contracts into `docs/terrain-collision-contract.md` or
   `docs/pair-manifest-terrain-params.md`; keep rationale here, not duplicate
   field definitions there.
