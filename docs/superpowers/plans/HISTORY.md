# Build History

Concise record of completed implementation milestones. Detailed step-by-step
plans live in git history. New plans use `YYYY-MM-DD-<slug>.md` in this
directory.

## Milestones

| Date | Theme | Outcome |
| --- | --- | --- |
| 2026-04-21 | Blind terrain tracking phase 1 | Downstream `mjlab` package, `TT-Tracking-TerrainBlind-Unitree-G1`, pair manifest runtime, paired mesh terrain injection |
| 2026-04-22 | Pair conversion | `terrain_tracking.convert_pair`, pair bundle layout (`pair.json`, `meta.json`, `motion.npz`), `scripts/train.sh` / `scripts/play.sh` |
| 2026-04-22 | Upstream mjlab dependency | PyPI `mjlab[cu128]` dependency; no local path fork |
| 2026-04-27 | Hfield collision manifest | Upstream `terrain_hf.npy` + `terrain_collision.json`; downstream manifest parsing and hfield debug backend |
| 2026-04-28 | Primitive box tile backend | `primitive_box_terrain.py`, default `primitive_boxes` backend, single-tile `TerrainGeneratorCfg` path in `apply_pair.py` |
| 2026-05-12 | Oracle terrain tracking | `TT-Tracking-TerrainOracleHeight-Unitree-G1` and `TT-Tracking-TerrainOracleTeacher-Unitree-G1` |
| 2026-05-15 | PHP teacher observation alignment | Pelvis-centric OracleTeacher observation contract while keeping `torso_link` tracking anchor |
| 2026-05-19 | OmniRetarget robot-terrain | `convert_omniretarget_robot_terrain`, `omniretarget_boxes` collision backend |

## Key Entry Points

- Converters: `src/terrain_tracking/convert_pair.py`,
  `src/terrain_tracking/convert_omniretarget_robot_terrain.py`
- Pair runtime: `src/terrain_tracking/runtime/apply_pair.py`
- Blind task: `src/terrain_tracking/tasks/blind_terrain_tracking/`
- Oracle task: `src/terrain_tracking/tasks/oracle_terrain_tracking/`
- Launch scripts: `scripts/train.sh`, `scripts/play.sh`, `scripts/exp/`

## Lifecycle

1. Add a dated plan file here while implementing a feature.
2. When the feature ships, add one row to the milestone table above.
3. Delete or archive the detailed plan; git retains the full history.
4. Move stable runtime contracts into `docs/terrain-collision-contract.md`,
   `docs/pair-manifest-terrain-params.md`, or `README.md` as appropriate.
5. Track unfinished terrain-collision work in `docs/terrain-collision-status.md`.
