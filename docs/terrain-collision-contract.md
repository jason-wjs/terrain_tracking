# Terrain Collision Contract

PARC terrain must not be imported into MuJoCo as one non-convex OBJ collision mesh. MuJoCo mesh collision uses convex hull behavior for mesh geoms, so a multi-box OBJ can look correct visually while its collision surface is bridged or filled in.

The training path should use an engine-friendly collision manifest:

- `terrain_hf.npy`: original PARC `terrain_data.hf` array.
- `terrain_collision.json`: collision metadata for MuJoCo hfield import.
- `multi_boxes.obj`: legacy retargeting asset and optional visual/debug asset, not the collision source for terrain_tracking when a collision manifest exists.

## Manifest Fields

```json
{
  "schema_version": 1,
  "terrain_name": "platform_001",
  "collision": {
    "type": "heightfield",
    "hf_file": "terrain_hf.npy",
    "min_point": [-1.0, -1.0],
    "dx": 0.4,
    "base_z": -0.32592592592592595,
    "xy_scale": 0.8148148148148148,
    "height_scale": 0.8148148148148148
  },
  "visual": {
    "file": "multi_boxes.obj",
    "role": "visual_only"
  }
}
```

Interpretation:

```text
world_x = xy_scale * (min_point[0] + i * dx)
world_y = xy_scale * (min_point[1] + j * dx)
world_z = height_scale * hf[i, j]
```

Scale is produced by the upstream retargeting repo from its actual config. `terrain_tracking` only reads `xy_scale` and `height_scale`; it does not import holosoma or recompute robot/human scale.

## Pair Manifest

`pair.json` keeps `terrain_file` for compatibility, and adds:

```json
{
  "terrain_collision_file": "/abs/path/to/terrain_collision.json",
  "terrain_visual_file": "/abs/path/to/multi_boxes.obj"
}
```

When `terrain_collision_file` is present, terrain_tracking uses it as source
terrain data and does not use the OBJ mesh as the default collision source.

## Runtime Collision Backends

`terrain_hf.npy` is source terrain data, not a mandate to use MuJoCo `hfield`
collision.

terrain_tracking supports these runtime backends:

- `primitive_boxes`: default for PARC platform/block terrain. Converts
  piecewise-constant hf regions into merged MuJoCo box geoms and places them in
  a shared terrain tile.
- `hfield`: debug backend. Uses MuJoCo hfield collision directly.
- `mesh`: legacy compatibility backend. Uses OBJ mesh collision and is not
  recommended for non-convex terrain.

For large training, terrain geoms should scale with unique terrain tiles, not
with `num_envs`.

## Defaults And Decisions

When `terrain_collision_file` is present:

- Default runtime backend: `primitive_boxes`.
- Collision scale comes from the manifest's `xy_scale` and `height_scale`, not
  from legacy `pair.json` mesh fields. See `pair-manifest-terrain-params.md`.
- `multi_boxes.obj` is visual/debug only on this path.

Backend selection:

| Backend | Role |
| --- | --- |
| `primitive_boxes` | Default PARC training path; one shared terrain tile via mjlab generator |
| `hfield` | Debug only; do not use as default PARC training backend |
| `mesh` | Legacy fallback when no collision manifest exists |
| `omniretarget_boxes` | OmniRetarget URDF terrain; separate input contract from PARC manifests |

Unstable work, open risks, and rationale are tracked in
`terrain-collision-status.md`.
