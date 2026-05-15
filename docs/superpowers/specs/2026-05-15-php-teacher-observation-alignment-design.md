# PHP Teacher Observation Alignment Design

## Goal

Align `TT-Tracking-TerrainOracleTeacher-Unitree-G1` with the teacher policy observation contract described in `docs/PHP.pdf`, while keeping the existing mjlab tracking task behavior stable.

The target PHP teacher observations are:

- Reference joint position and velocity.
- Reference pelvis pose error.
- Pelvis linear and angular velocity.
- Current joint position and velocity.
- Previous action.
- A `0.7 m x 0.7 m` height scan.
- Privileged pelvis global position and velocity.

This change is limited to the OracleTeacher observation contract. It does not change rewards, terminations, reset behavior, PPO configuration, the blind task, or the OracleHeight task.

## Current State

The existing G1 tracking task uses `torso_link` as `MotionCommandCfg.anchor_body_name`. As a result, the existing anchor observations are torso-anchor observations:

- `motion_anchor_pos_b`
- `motion_anchor_ori_b`
- `global_anchor_pos_error_w`
- `global_anchor_lin_vel_error_w`
- `reference_anchor_lin_vel_w`

The motion command still includes `pelvis` in `body_names`, and mjlab exposes both reference and robot body world-frame states through `MotionCommand`. Therefore pelvis-specific observations can be added without changing the motion data format or the tracking anchor.

## Design

Keep `anchor_body_name="torso_link"` for the underlying tracking task. This preserves existing reward, termination, reset, visualization, and tracking-metric semantics.

For `TT-Tracking-TerrainOracleTeacher-Unitree-G1`, replace teacher-facing torso/base observation terms with pelvis-specific observation terms:

- Keep `command` as the reference joint position and velocity observation.
- Replace `motion_anchor_pos_b` with `reference_pelvis_pos_error_b`.
- Replace `motion_anchor_ori_b` with `reference_pelvis_ori_error_b`.
- Replace `base_lin_vel` with `pelvis_lin_vel`.
- Replace `base_ang_vel` with `pelvis_ang_vel`.
- Keep `joint_pos`, `joint_vel`, and `actions`.
- Keep `height_scan`.

Replace the current anchor privileged terms with PHP-style pelvis privileged terms:

- Remove `global_anchor_pos_error_w`.
- Remove `global_anchor_lin_vel_error_w`.
- Remove `reference_anchor_lin_vel_w`.
- Add `pelvis_global_pos_w`.
- Add `pelvis_global_lin_vel_w`.

The pelvis-specific functions will live in `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/observations.py`. They will find the `pelvis` index from `command.cfg.body_names` and read from the existing `MotionCommand` tensors:

- Reference pelvis pose: `command.body_pos_w[:, pelvis_idx]`, `command.body_quat_w[:, pelvis_idx]`.
- Robot pelvis pose: `command.robot_body_pos_w[:, pelvis_idx]`, `command.robot_body_quat_w[:, pelvis_idx]`.
- Robot pelvis velocity: `command.robot_body_lin_vel_w[:, pelvis_idx]`, `command.robot_body_ang_vel_w[:, pelvis_idx]`.

`reference_pelvis_pos_error_b` and `reference_pelvis_ori_error_b` should use the same transform convention as mjlab's existing `motion_anchor_pos_b` and `motion_anchor_ori_b`: express the reference pelvis pose relative to the current robot pelvis frame. Orientation should use the first two columns of the relative rotation matrix, matching mjlab's existing anchor orientation encoding.

`pelvis_global_pos_w` and `pelvis_global_lin_vel_w` return the current robot pelvis world-frame position and linear velocity. They intentionally do not return tracking error; this follows the PHP paper wording, "pelvis global position and velocity".

## Height Scan

Keep the raycast sensor frame on `torso_link`.

PHP specifies the height scan size, but does not specify the body frame used to mount it. The current torso-mounted scan is already part of `TerrainOracleHeight`, is stable in the existing project, and avoids expanding this change beyond teacher observation alignment. The teacher proprioceptive and privileged observations become pelvis-centric, while terrain scanning remains unchanged.

## Actor And Critic

Apply the new OracleTeacher observation contract to both actor and critic groups. OracleTeacher is a simulation-only teacher policy, so the actor is allowed to receive privileged pelvis global information.

`TerrainOracleHeight` continues to receive only the height scan extension over the blind task. It must not receive the pelvis-only teacher terms.

## Tests

Update `tests/test_oracle_terrain_tracking_config.py` to verify:

- Blind and OracleHeight configs do not include pelvis-only teacher terms.
- OracleTeacher actor and critic include:
  - `reference_pelvis_pos_error_b`
  - `reference_pelvis_ori_error_b`
  - `pelvis_lin_vel`
  - `pelvis_ang_vel`
  - `pelvis_global_pos_w`
  - `pelvis_global_lin_vel_w`
- OracleTeacher actor and critic no longer include:
  - `motion_anchor_pos_b`
  - `motion_anchor_ori_b`
  - `base_lin_vel`
  - `base_ang_vel`
  - `global_anchor_pos_error_w`
  - `global_anchor_lin_vel_error_w`
  - `reference_anchor_lin_vel_w`
- Pelvis observation functions select the `pelvis` body index, not the tracking anchor index.
- Pelvis observation functions return tensors with shape `[num_envs, 3]`, except the orientation encoding which should match the existing orientation term shape.
- A clear error is raised if `pelvis` is missing from `command.cfg.body_names`.

Run the focused config tests after implementation:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_oracle_terrain_tracking_config.py
```

Run the oracle smoke tests if the focused tests pass:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_env_smoke.py -k oracle_terrain
```

## Non-Goals

- Do not change `MotionCommandCfg.anchor_body_name` from `torso_link` to `pelvis`.
- Do not change reward terms, termination thresholds, reset behavior, adaptive sampling, PPO hyperparameters, or training scripts.
- Do not change `TerrainOracleHeight`.
- Do not add a second pelvis-mounted height scan variant in this pass.

