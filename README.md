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

When `terrain_collision_file` is present, `multi_boxes.obj` is not used as the collision source. See `docs/terrain-collision-contract.md`.

## Oracle Perception Tasks

This package also registers oracle perception tracking tasks for the current height-map stage:

- `TT-Tracking-TerrainOracleHeight-Unitree-G1`: blind tracking observations plus a clean `0.7 m x 0.7 m` yaw-aligned terrain height scan from `torso_link`.
- `TT-Tracking-TerrainOracleTeacher-Unitree-G1`: oracle height observations plus PHP teacher-style global tracking error terms for `torso_link`.

Both tasks reuse the blind tracking rewards, terminations, terrain pair application, and PPO defaults. They are simulation-only oracle tasks and are not deployment policies.

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
