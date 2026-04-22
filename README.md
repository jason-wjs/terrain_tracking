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
