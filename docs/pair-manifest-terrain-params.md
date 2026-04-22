# Pair Manifest Terrain Parameters

This note documents the three terrain correction parameters in `pair.json`:

- `terrain_translation`
- `terrain_quat_xyzw`
- `terrain_scale`

These fields exist so the terrain mesh used by `terrain_tracking` can be aligned with the coordinate frame and physical scale assumed by the paired `motion.npz`.

## `terrain_scale`

`terrain_scale` is applied directly to the terrain mesh vertices when `terrain_tracking` loads the paired terrain. In other words, it is not metadata-only; it changes the geometry that ends up in the simulation.

This matters for the current PARC to G1 pipeline:

- upstream `holosoma/parc_process` exports an original terrain mesh as `multi_boxes.obj`
- the retargeting path may then generate scaled `URDF` and `XML` wrappers, for example `multi_boxes_scaled_0.74_0.74_0.74.urdf`
- the retargeted motion is optimized against that scaled terrain, not necessarily against the raw unscaled `.obj`

At the moment, `terrain_tracking` consumes the terrain through a mesh file path and explicit transform parameters in `pair.json`. The terrain path is expected to be a mesh file that can be loaded as geometry; in our current workflow this means using the exported `.obj`, not the upstream `URDF` or MuJoCo `XML` wrappers.

Therefore:

- if `terrain_file` points to the original unscaled `multi_boxes.obj`, `terrain_scale` must carry the scale used upstream during retargeting
- if a separately baked scaled mesh is provided as the terrain file, then `terrain_scale` should usually stay at `1 1 1`

For example, for the current `platform_001` PARC sample, the upstream retargeting path uses a scale of approximately `0.7415730337078652` on all three axes, so passing `terrain_scale = [1, 1, 1]` with the raw `multi_boxes.obj` would not match the terrain used during retargeting.
