# Agent instructions for `terrain_tracking`

## Dataset locations

Durable training data lives under `/home/humanoid/Downloads/Data/`.

| Dataset | Directory | Use in this repo |
|---------|-----------|------------------|
| **PARC** | `parc_initial_aug_g1` | All PARC pair data (motions, terrains, `terrain_collision.json`). Convert with `convert_pair.py` / `scripts/convert.sh`; train with `--collision-backend primitive_boxes` (or `hfield` / `mesh`). Typical paired root: `parc_initial_aug_g1/parc_process/paired`. |
| **OmniRetarget** | `tt_converted_omniretarget` | Pre-converted pair bundles (`motion.npz`, `pair.json`, `meta.json`) from OmniRetarget `robot-terrain` clips. One subdirectory per sample (e.g. `climb_00_z_scale_1.0/`). Convert raw clips with `convert_omniretarget_robot_terrain.py`; train with `--collision-backend omniretarget_boxes`. |

## Oracle teacher training (mid_blocks)

- **General teacher** (mid_blocks dataset, in progress): `logs/rsl_rl/tt_general_pair_dataset_mid_blocks/2026-05-27_19-08-29_mid_blocks_general_g1_oracle_teacher_n8192_adaptive`
- **Single-pair teacher** (mid_blocks `004_dm`): `logs/rsl_rl/tt_single_pair_mid_blocks_004_dm/2026-05-15_14-26-56_mid_blocks_004_dm_g1_oracle_teacher_n8192_adaptive`

