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
