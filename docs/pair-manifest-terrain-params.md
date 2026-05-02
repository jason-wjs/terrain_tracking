# Pair Manifest Terrain Parameters

This note documents the legacy mesh correction parameters in `pair.json`:

- `terrain_translation`
- `terrain_quat_xyzw`
- `terrain_scale`

These fields exist so the legacy terrain mesh fallback can be aligned with the coordinate frame and physical scale assumed by the paired `motion.npz`.

For PARC heightfield terrain, prefer `terrain_collision_file`. When it exists,
`terrain_tracking` uses `terrain_collision.json` and `terrain_hf.npy` as source
terrain data. The default runtime backend is `primitive_boxes`; the `hfield`
backend remains available explicitly for diagnostics. In both paths, collision
scale comes from the collision manifest's `xy_scale` and `height_scale`, not
from `terrain_scale`.

## `terrain_scale`

`terrain_scale` is applied directly to the terrain mesh vertices when `terrain_tracking` loads the paired terrain. In other words, it is not metadata-only; it changes the geometry that ends up in the simulation.

This matters for the legacy PARC OBJ mesh path:

- upstream `holosoma/parc_process` exports an original terrain mesh as `multi_boxes.obj`
- the retargeting path may then generate scaled `URDF` and `XML` wrappers
- the retargeted motion is optimized against that scaled terrain, not necessarily against the raw unscaled `.obj`

If `terrain_collision_file` is absent, `terrain_tracking` consumes the terrain through a mesh file path and explicit transform parameters in `pair.json`. The terrain path is expected to be a mesh file that can be loaded as geometry; in this fallback workflow this means using the exported `.obj`, not the upstream `URDF` or MuJoCo `XML` wrappers.

Therefore:

- if `terrain_file` points to the original unscaled `multi_boxes.obj`, `terrain_scale` must carry the scale used upstream during retargeting
- if a separately baked scaled mesh is provided as the terrain file, then `terrain_scale` should usually stay at `1 1 1`
- if `terrain_collision_file` is present, do not use `terrain_scale` to express collision scale; regenerate the upstream `terrain_collision.json` so it records the actual scale used by retargeting
