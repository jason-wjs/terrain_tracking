# terrain_tracking Context

## Domain Terms

**Pair manifest** — `pair.json` file that connects one tracking motion with one
terrain source. It is the runtime input contract for train/play entry points.

**Pair bundle** — Directory containing `pair.json`, `meta.json`, and any local
motion assets written by a converter.

**Terrain collision manifest** — `terrain_collision.json` file describing PARC
heightfield collision data and scale. It is preferred over legacy OBJ mesh
collision when present.

**Collision backend** — Runtime choice for turning a pair manifest terrain source
into MuJoCo collision geometry. Current values are `primitive_boxes`, `hfield`,
`mesh`, and `omniretarget_boxes`.

**Terrain application** — Runtime step that applies a pair manifest to an mjlab
environment config: set the motion file, configure terrain geometry, and tune
simulation contact capacity.

**OmniRetarget robot-terrain sample** — OmniRetarget motion clip plus a matching
terrain URDF under `models/terrain`. The converter turns it into a normal pair
bundle.

**Oracle perception task** — Simulation-only tracking task that augments blind
tracking observations with clean terrain or teacher observations. These are not
deployment policy tasks.

**Experiment preset** — Thin shell Adapter under `scripts/exp` that declares a
commonly used train/play configuration and delegates launch behavior to
`scripts/lib/common.sh`.

**Conversion preset** — Thin shell Adapter under `scripts/exp/convert` that
declares a common pair bundle conversion and delegates execution to
`scripts/convert.sh` through `scripts/lib/common.sh`.

## Stable Release Surface

- `terrain_tracking.convert_pair`
- `terrain_tracking.convert_omniretarget_robot_terrain`
- `terrain_tracking.runtime.apply_pair_manifest_to_env_cfg`
- `terrain_tracking.tasks.blind_terrain_tracking.scripts.train`
- `terrain_tracking.tasks.blind_terrain_tracking.scripts.play`
- `scripts/convert.sh`, `scripts/train.sh`, `scripts/play.sh`, and
  `scripts/exp/*/*.sh`

Historical build notes live under `docs/superpowers/`; they are not the release
interface.
