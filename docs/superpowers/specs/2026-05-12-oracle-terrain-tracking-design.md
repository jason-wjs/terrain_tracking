# Oracle Terrain Tracking Design

Date: 2026-05-12

## Goal

Add an oracle perception stage for terrain tracking while preserving the current blind tracking task as the baseline. This stage follows the PHP teacher-policy direction in `docs/PHP.pdf`: train motion-tracking policies that can access privileged simulation state and a clean local height scan. It does not target deployment, depth images, DAgger, or student distillation.

## Scope

In scope:

- Add a new `oracle_terrain_tracking` task family.
- Register two Unitree G1 task variants:
  - `TT-Tracking-TerrainOracleHeight-Unitree-G1`
  - `TT-Tracking-TerrainOracleTeacher-Unitree-G1`
- Reuse the current blind terrain tracking rewards, terminations, events, motion command setup, terrain pair application, and PPO hyperparameters.
- Add clean oracle observations to actor and critic.
- Verify task registration, observation composition, reset/step smoke, and short training smoke.

Out of scope:

- Depth images.
- Student policy or DAgger training.
- Observation noise, camera artifacts, latency, or deployment constraints.
- `No-State-Estimation` oracle variants.
- Changing blind task behavior.
- Long training runs or convergence claims.

## Task Architecture

Create a new task family:

```text
src/terrain_tracking/tasks/oracle_terrain_tracking/
  __init__.py
  config/
    __init__.py
    g1/
      __init__.py
      env_cfgs.py
      rl_cfg.py
      observations.py
```

`oracle_terrain_tracking.config.g1.env_cfgs` should call the existing `unitree_g1_blind_terrain_tracking_env_cfg(...)` first, then layer oracle-only observations on top. This keeps the blind task isolated and avoids duplicating the full tracking setup.

`rl_cfg.py` should reuse the blind PPO runner config and only change metadata such as `experiment_name`, for example to `g1_oracle_terrain_tracking`. Network sizes, PPO hyperparameters, `num_steps_per_env`, and `max_iterations` stay aligned with the blind config unless a later experiment shows a concrete need to change them.

`terrain_tracking._mjlab_tasks` must import the new oracle config module so both task IDs are registered through the existing `mjlab.tasks` entry point.

## Oracle Height Task

`TT-Tracking-TerrainOracleHeight-Unitree-G1` extends the blind actor and critic observations with a clean local height scan.

Sensor configuration:

- Sensor type: `RayCastSensorCfg`
- Sensor name: `terrain_scan`
- Frame: `ObjRef(type="body", name="torso_link", entity="robot")`
- Pattern: `GridPatternCfg(size=(0.7, 0.7), resolution=0.1)`
- Ray alignment: `yaw`
- Ray direction: default downward direction, `(0.0, 0.0, -1.0)`
- Included geom groups: `(0,)`, terrain only
- Max distance: `5.0`
- Debug visualization: optional, but default can match mjlab terrain scan behavior

Observation configuration:

- Observation function: `mjlab.envs.mdp.height_scan`
- Term name: `height_scan`
- Add to both `actor` and `critic`
- No noise
- No delay
- No corruption specific to this term
- Scale: `1 / max_distance`

Value semantics follow mjlab's native `height_scan`:

```text
height = torso_link_z - terrain_hit_z
```

This is a vertical clearance scan from the yaw-aligned torso frame, not a custom `terrain_z` map. With the current mjlab `GridPatternCfg` behavior, `size=(0.7, 0.7)` and `resolution=0.1` are expected to produce approximately `8 x 8 = 64` rays. Tests should lock the actual observed dimension instead of assuming a hand-written formula.

## Oracle Teacher Task

`TT-Tracking-TerrainOracleTeacher-Unitree-G1` extends `TerrainOracleHeight` with PHP teacher-style global tracking privileged observations. The terms use the existing `MotionCommand` anchor body, which is currently `torso_link`, rather than introducing a separate pelvis/root convention.

Add the following terms to both `actor` and `critic`:

- `global_anchor_pos_error_w`
  - Definition: `command.anchor_pos_w - command.robot_anchor_pos_w`
  - Shape: `[num_envs, 3]`
  - Purpose: world-frame anchor position tracking error for recovery.
- `global_anchor_lin_vel_error_w`
  - Definition: `command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w`
  - Shape: `[num_envs, 3]`
  - Purpose: world-frame anchor velocity tracking error for recovery.
- `reference_anchor_lin_vel_w`
  - Definition: `command.anchor_lin_vel_w`
  - Shape: `[num_envs, 3]`
  - Purpose: reference global anchor velocity context.

These terms should be implemented in `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/observations.py`. The existing mjlab `MotionCommand` already exposes `anchor_pos_w`, `robot_anchor_pos_w`, `anchor_lin_vel_w`, and `robot_anchor_lin_vel_w`, so no finite-difference helper is required.

The task intentionally avoids feeding absolute robot world position. The privileged state uses tracking-error semantics to support recovery without encouraging policies to memorize one terrain's absolute coordinates.

## Training And Play Entry Points

Reuse the existing blind training and play scripts because they already support arbitrary registered task IDs:

```bash
uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.train \
  --pair-manifest /path/to/pair.json \
  --task TT-Tracking-TerrainOracleHeight-Unitree-G1
```

```bash
uv run python -m terrain_tracking.tasks.blind_terrain_tracking.scripts.train \
  --pair-manifest /path/to/pair.json \
  --task TT-Tracking-TerrainOracleTeacher-Unitree-G1
```

`apply_pair_manifest_to_env_cfg(...)` remains the shared terrain/motion pair application path. It is task-agnostic as long as the environment config contains the `motion` command and terrain scene fields, which the oracle tasks inherit from the blind task.

The root `scripts/train.sh` default task does not need to change. Users can pass the oracle task with `--task` or the script's existing task override mechanism. README should show at least one oracle task training example.

## Testing And Acceptance

Unit tests:

- Registry:
  - `TT-Tracking-TerrainOracleHeight-Unitree-G1` is registered.
  - `TT-Tracking-TerrainOracleTeacher-Unitree-G1` is registered.
- Config composition:
  - `TerrainOracleHeight` actor and critic include `height_scan`.
  - `TerrainOracleTeacher` actor and critic include `height_scan`, `global_anchor_pos_error_w`, `global_anchor_lin_vel_error_w`, and `reference_anchor_lin_vel_w`.
  - Existing blind task actor and critic do not gain oracle-only terms.
- Observation functions:
  - Teacher privileged observation functions return tensors with shape `[num_envs, 3]`.
- Environment smoke:
  - Both oracle tasks can build with a small paired manifest.
  - Both oracle tasks can `reset()` and `step()`.
  - Actor observation dimension is larger than the blind actor observation dimension.
- CLI compatibility:
  - Existing train/play help remains valid.

Implementation-stage acceptance:

- Both oracle tasks pass registration and env smoke tests.
- A short training smoke run works for both oracle tasks, for example 100-500 iterations, without requiring convergence.
- Old blind checkpoints are not expected to load into oracle tasks because actor observation dimensions change.

## Risks And Constraints

- Terrain raycasting assumes terrain geoms are in geom group `0`. Tests should lock this against the current primitive box terrain backend.
- `height_scan` returns `max_distance` for misses. Oracle perception does not simulate miss artifacts, but misses can still occur if the scan footprint extends outside terrain coverage. Training pair terrain should cover the local scan footprint.
- Adding privileged observations to actor and critic changes the learned policy contract. These tasks are not deployment policies.
- This design preserves current rewards and PPO settings. If later training shows instability, tune in a follow-up plan rather than expanding this initial oracle-perception scope.
