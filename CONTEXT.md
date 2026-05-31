# terrain_tracking Context

## Domain Terms

**Pair manifest** — `pair.json` file that connects one tracking motion with one
terrain source. It is the runtime input contract for train/play entry points.

**Pair bundle** — Directory containing `pair.json`, `meta.json`, and any local
motion assets written by a converter.

**Pair dataset** — Collection of motion-terrain pairing records used to train a
general controller across many terrain-aware tracking scenarios.

**Pair ID** — Stable identifier for one record inside a pair dataset; it is the
sampling key that keeps the motion clip and terrain tile aligned during reset.

**Pair-local frame** — Local coordinate frame shared by the motion trajectory
and terrain geometry of one motion-terrain pair.

**Paired reset** — Environment reset that samples one pair ID and uses it to
select both the motion clip and the terrain tile for that episode.

**Pair-frame sampler** — Reset-time sampler that chooses both the pair ID and
the start frame or time bin for that pair.

**Terrain adapter** — Source-specific translator that turns a pair dataset
terrain reference into a common terrain tile.

**Terrain tile** — Placement-ready terrain geometry for one pair dataset record,
with a stable origin used to align motion replay and terrain contact.

**Terrain tile bank** — Compiled collection of terrain tiles where each pair ID
maps to one tile origin.

**Tile packing** — Deterministic placement of terrain tiles in world space using
inflated pair-local bounds so neighboring pairs do not interact during normal
episodes.

**Tile bounds violation** — Failure condition where a robot leaves the inflated
pair-local bounds of its currently sampled pair.

**Tile origin** — World-space translation of one pair-local frame inside a
terrain tile bank; it is not the terrain center or terrain bounds corner.

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

Historical build notes live under `docs/superpowers/`; they are not the release
interface.
