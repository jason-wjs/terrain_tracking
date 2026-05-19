# OmniRetarget Robot-Terrain Single Pair Design

## Goal

Support importing one OmniRetarget `robot-terrain` sample into `terrain_tracking` as a normal single-pair training input.

This stage must prove a minimal end-to-end path:

- read one OmniRetarget motion clip with `qpos` and `fps`
- convert it into the mjlab tracking motion format used by `MotionCommand`
- load the matching OmniRetarget terrain URDF as primitive MuJoCo box collision
- write a pair bundle with `motion.npz`, `pair.json`, and `meta.json`
- build/reset/step the current terrain tracking task with that pair
- run a short training smoke

This stage does not add a multi-pair dataset sampler and does not train across all 145 OmniRetarget samples.

## Source Data Contract

An OmniRetarget `robot-terrain` sample is a motion clip plus external terrain assets. The `.npz` does not contain terrain geometry.

Example input:

```text
robot-terrain/climb_00_z_scale_1.0.npz
models/terrain/climb_00/multi_boxes_z_scale_1.0.urdf
models/terrain/climb_00/box_models/box1.obj
```

The motion file contains:

- `qpos`: shape `[T, 36]`
  - columns `0:4`: root quaternion in `wxyz`
  - columns `4:7`: root position in world frame
  - columns `7:36`: 29 Unitree G1 joint positions
- `fps`: source sample rate, expected to be 30 for the published robot-terrain clips
- optional `human_joints`: ignored by this feature

The terrain file is a URDF whose mesh tags reference one or more box OBJ files under `box_models/`. The URDF mesh scale is part of the terrain geometry and must be applied.

## Runtime Representation

The pair remains a normal `pair.json` bundle. The existing `terrain_file` field points to the OmniRetarget terrain URDF. The training command selects the new runtime backend:

```bash
uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.train \
  --pair-manifest /tmp/tt_converted_omniretarget/climb_00_z_scale_1.0/pair.json \
  --collision-backend omniretarget_boxes \
  --task TT-Tracking-TerrainOracleTeacher-Unitree-G1
```

`omniretarget_boxes` is distinct from the existing `primitive_boxes` backend:

- `primitive_boxes` means `terrain_collision.json` plus `terrain_hf.npy` converted from PARC-style heightfields.
- `omniretarget_boxes` means OmniRetarget terrain URDF plus box OBJ files converted directly to MuJoCo box geoms.

Both backends may use MuJoCo primitive box geoms internally, but their input contracts are different.

## Motion Conversion

The converter writes mjlab tracking motion arrays:

- `fps`: output sample rate, fixed at 50
- `joint_pos`: shape `[T_out, 29]`
- `joint_vel`: shape `[T_out, 29]`
- `body_pos_w`: shape `[T_out, nbody, 3]`
- `body_quat_w`: shape `[T_out, nbody, 4]`, `wxyz`
- `body_lin_vel_w`: shape `[T_out, nbody, 3]`
- `body_ang_vel_w`: shape `[T_out, nbody, 3]`

The current task advances one motion frame per control step. Its simulation timestep is 0.005 seconds with decimation 4, so the control step is 0.02 seconds. Therefore OmniRetarget 30 Hz clips must be resampled to 50 Hz before training.

Conversion uses the current mjlab Unitree G1 MJCF model as the authority for forward kinematics. For each output frame:

1. Fill MuJoCo `qpos` as `[x, y, z, qw, qx, qy, qz, joint_pos...]`.
2. Run `mj_forward`.
3. Read all body world positions and quaternions.
4. Compute joint and body velocities from adjacent output frames.

This avoids relying on the OmniRetarget visualization URDF for training kinematics.

## Terrain Conversion

The backend parses the terrain URDF and creates collision geoms from every unique collision mesh reference:

1. Read each collision `<mesh filename="..." scale="sx sy sz">`.
2. Resolve filenames relative to the URDF directory.
3. Load the OBJ vertices.
4. Apply mesh scale.
5. Fit the scaled vertices as a box primitive.
6. Add a MuJoCo box geom with the fitted position, half-size, and yaw orientation.

The published robot-terrain obstacles are box meshes, often rotated in the XY plane. The backend should preserve yaw rotation instead of axis-aligning every box. If a mesh cannot be represented as a single box within tolerance, the backend raises a clear error and the converter does not silently approximate it.

The backend also adds a ground/base box because OmniRetarget visualization loads a separate ground model, while the terrain tracking task disables the default plane collision. The ground footprint should cover the terrain obstacle bounds and the motion root path bounds for the converted sample.

## Pair Bundle Conversion

The new converter command accepts one motion file and finds the matching terrain URDF from the sample name:

```bash
uv run python -m terrain_tracking.convert_omniretarget_robot_terrain \
  --motion-file /path/to/OmniRetarget_Dataset/robot-terrain/climb_00_z_scale_1.0.npz \
  --terrain-root /path/to/OmniRetarget_Dataset/models/terrain \
  --output-dir /tmp/tt_converted_omniretarget \
  --sample-name climb_00_z_scale_1.0
```

The output layout is:

```text
/tmp/tt_converted_omniretarget/climb_00_z_scale_1.0/
  motion.npz
  pair.json
  meta.json
```

The converter records source paths, source fps, output fps, frame counts, and terrain URDF path in `meta.json`.

## Testing And Verification

Unit tests cover:

- OmniRetarget sample-name to terrain-URDF resolution.
- `qpos` parsing and validation.
- 30 Hz to 50 Hz resampling frame counts.
- output motion `.npz` schema and shapes.
- URDF mesh parsing and scale handling.
- rotated box fitting from OBJ vertices.
- `omniretarget_boxes` backend registration in `apply_pair_manifest_to_env_cfg`.

Smoke verification covers:

- build/reset/step one environment from a generated OmniRetarget pair bundle
- run a short training smoke with `num_envs=1` and a tiny iteration count

## Out Of Scope

- Multi-pair manifest format.
- Sampling across all OmniRetarget robot-terrain clips during one run.
- Rewriting PARC `primitive_boxes`.
- Converting OmniRetarget terrain to PARC heightfield manifests.
- Object or object-terrain OmniRetarget subsets.
