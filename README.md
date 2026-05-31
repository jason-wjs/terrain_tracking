# terrain_tracking

Downstream `mjlab` extension package for blind terrain tracking experiments.

## Quick Start

Train with a paired motion/terrain manifest:

```bash
bash scripts/train.sh --pair-manifest /abs/path/to/pair.json
```

Play a paired motion in the terrain scene with a dummy policy:

```bash
bash scripts/play.sh --pair-manifest /abs/path/to/pair.json --agent zero
```

Reusable shell launchers live under `scripts/`:

- `scripts/convert.sh`, `scripts/train.sh`, and `scripts/play.sh` are the stable launch entry points.
- `scripts/exp/convert/*.sh`, `scripts/exp/train/*.sh`, and `scripts/exp/play/*.sh` are one-command presets.
- `scripts/lib/common.sh` holds the shared launch implementation used by all shell entry points.

Create a pair bundle that uses heightfield collision:

```bash
uv run python -m terrain_tracking.convert_pair \
  --motion-file /tmp/tt_converted/platform_001/motion.npz \
  --terrain-file /tmp/parc_process_workspace/workspace/platform_001/multi_boxes.obj \
  --terrain-collision-file /tmp/parc_process_workspace/workspace/platform_001/terrain_collision.json \
  --terrain-visual-file /tmp/parc_process_workspace/workspace/platform_001/multi_boxes.obj \
  --output-dir /tmp/tt_converted \
  --sample-name platform_001
```

The same conversion is available through the shell entry point:

```bash
CONVERT_KIND=pair \
MOTION_FILE=/tmp/tt_converted/platform_001/motion.npz \
TERRAIN_FILE=/tmp/parc_process_workspace/workspace/platform_001/multi_boxes.obj \
TERRAIN_COLLISION_FILE=/tmp/parc_process_workspace/workspace/platform_001/terrain_collision.json \
TERRAIN_VISUAL_FILE=/tmp/parc_process_workspace/workspace/platform_001/multi_boxes.obj \
SAMPLE_NAME=platform_001 \
bash scripts/convert.sh
```

When `terrain_collision_file` is present, `multi_boxes.obj` is not used as the collision source. See [terrain collision contract](docs/terrain-collision-contract.md) and [status](docs/terrain-collision-status.md).

Pair manifest terrain parameters are documented in [pair-manifest-terrain-params.md](docs/pair-manifest-terrain-params.md). Repository domain terms are defined in `CONTEXT.md`.

## OmniRetarget Robot-Terrain Samples

Convert one OmniRetarget `robot-terrain` sample into a regular pair bundle:

```bash
uv run python -m terrain_tracking.convert_omniretarget_robot_terrain \
  --motion-file /path/to/OmniRetarget_Dataset/robot-terrain/climb_00_z_scale_1.0.npz \
  --terrain-root /path/to/OmniRetarget_Dataset/models/terrain \
  --output-dir /tmp/tt_converted_omniretarget \
  --sample-name climb_00_z_scale_1.0
```

The shell entry point supports the OmniRetarget converter too:

```bash
CONVERT_KIND=omniretarget_robot_terrain \
MOTION_FILE=/path/to/OmniRetarget_Dataset/robot-terrain/climb_00_z_scale_1.0.npz \
TERRAIN_ROOT=/path/to/OmniRetarget_Dataset/models/terrain \
SAMPLE_NAME=climb_00_z_scale_1.0 \
bash scripts/convert.sh
```

The converter writes:

```text
/tmp/tt_converted_omniretarget/climb_00_z_scale_1.0/
  motion.npz
  pair.json
  meta.json
```

Train or play the converted pair with the OmniRetarget box backend:

```bash
uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.train \
  --pair-manifest /tmp/tt_converted_omniretarget/climb_00_z_scale_1.0/pair.json \
  --collision-backend omniretarget_boxes \
  --task TT-Tracking-TerrainOracleHeight-Unitree-G1 \
  --env.scene.num-envs 1 \
  --agent.max-iterations 1
```

`omniretarget_boxes` reads the OmniRetarget terrain URDF and creates MuJoCo
primitive box collision geoms. It does not use the PARC heightfield
`terrain_collision.json` path.

## Oracle Perception Tasks

This package also registers oracle perception tracking tasks for the current height-map stage:

- `TT-Tracking-TerrainOracleHeight-Unitree-G1`: blind tracking observations plus a clean `0.7 m x 0.7 m` yaw-aligned terrain height scan from `torso_link`.
- `TT-Tracking-TerrainOracleTeacher-Unitree-G1`: oracle height observations plus PHP teacher-style global tracking error terms for `torso_link`.

Both tasks reuse the blind tracking rewards, terminations, terrain pair application, and PPO defaults. They are simulation-only oracle tasks and are not deployment policies.

### General Teacher Known Issues

- `TT-Tracking-TerrainOracleTeacherGeneral-Unitree-G1` currently only supports the intended `sampling_mode=adaptive` path for multi-pair training. `sampling_mode=start` is a known bug: it resets all envs to pair index 0 instead of sampling across pairs from frame 0.

Train with the existing paired training entry point:

```bash
uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.train \
  --pair-manifest /abs/path/to/pair.json \
  --task TT-Tracking-TerrainOracleHeight-Unitree-G1
```

```bash
uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.train \
  --pair-manifest /abs/path/to/pair.json \
  --task TT-Tracking-TerrainOracleTeacher-Unitree-G1
```

## Development Checks

```bash
uv run ruff check .
uv run pyright
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` keeps unrelated globally installed pytest
plugins out of the test run.
