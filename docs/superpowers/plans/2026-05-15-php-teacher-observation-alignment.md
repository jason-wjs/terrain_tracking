# PHP Teacher Observation Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the OracleTeacher observation contract with PHP teacher-policy observations by replacing torso-anchor teacher terms with pelvis-specific observations while preserving the existing `torso_link` tracking anchor.

**Architecture:** Add pelvis-specific observation functions in the oracle G1 observation module, then have only `TT-Tracking-TerrainOracleTeacher-Unitree-G1` replace inherited torso-anchor/base terms with those pelvis terms. Keep `TerrainOracleHeight`, rewards, terminations, motion anchor configuration, height scan frame, and PPO config unchanged.

**Tech Stack:** Python 3.13, PyTorch tensors, mjlab `MotionCommand`, mjlab observation config terms, pytest.

---

## File Structure

- Modify `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/observations.py`
  - Owns OracleTeacher-only observation functions.
  - Add pelvis body lookup and pelvis-specific observation functions.
- Modify `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/env_cfgs.py`
  - Owns OracleHeight and OracleTeacher observation wiring.
  - Replace inherited torso/base terms only in OracleTeacher.
- Modify `tests/test_oracle_terrain_tracking_config.py`
  - Owns unit coverage for oracle observation configuration and helper functions.
  - Update expected teacher terms and fake motion command fixtures.

## Task 1: Lock The New OracleTeacher Observation Contract With Tests

**Files:**
- Modify: `tests/test_oracle_terrain_tracking_config.py`
- Test: `tests/test_oracle_terrain_tracking_config.py`

- [ ] **Step 1: Replace old teacher term constants**

In `tests/test_oracle_terrain_tracking_config.py`, replace the existing `ORACLE_TEACHER_TERMS` constant with these constants:

```python
ORACLE_TEACHER_TERMS = {
  "reference_pelvis_pos_error_b",
  "reference_pelvis_ori_error_b",
  "pelvis_lin_vel",
  "pelvis_ang_vel",
  "pelvis_global_pos_w",
  "pelvis_global_lin_vel_w",
}

REPLACED_TEACHER_TERMS = {
  "motion_anchor_pos_b",
  "motion_anchor_ori_b",
  "base_lin_vel",
  "base_ang_vel",
  "global_anchor_pos_error_w",
  "global_anchor_lin_vel_error_w",
  "reference_anchor_lin_vel_w",
}
```

- [ ] **Step 2: Replace the fake motion command fixture**

Replace `_FakeMotionCommand` in `tests/test_oracle_terrain_tracking_config.py` with:

```python
class _FakeMotionCommand:
  def __init__(self, body_names: tuple[str, ...] = ("torso_link", "pelvis")) -> None:
    self.cfg = type("Cfg", (), {"body_names": body_names})()
    self.body_pos_w = torch.tensor(
      [
        [[10.0, 10.0, 10.0], [1.0, 2.0, 3.0]],
        [[20.0, 20.0, 20.0], [4.0, 5.0, 6.0]],
      ],
      dtype=torch.float32,
    )
    self.body_quat_w = torch.tensor(
      [
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
      ],
      dtype=torch.float32,
    )
    self.robot_body_pos_w = torch.tensor(
      [
        [[0.0, 0.0, 0.0], [0.5, 1.5, 2.5]],
        [[0.0, 0.0, 0.0], [3.5, 4.5, 5.5]],
      ],
      dtype=torch.float32,
    )
    self.robot_body_quat_w = torch.tensor(
      [
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
      ],
      dtype=torch.float32,
    )
    self.robot_body_lin_vel_w = torch.tensor(
      [
        [[9.0, 9.0, 9.0], [0.1, 0.2, 0.3]],
        [[8.0, 8.0, 8.0], [0.4, 0.5, 0.6]],
      ],
      dtype=torch.float32,
    )
    self.robot_body_ang_vel_w = torch.tensor(
      [
        [[7.0, 7.0, 7.0], [1.1, 1.2, 1.3]],
        [[6.0, 6.0, 6.0], [1.4, 1.5, 1.6]],
      ],
      dtype=torch.float32,
    )
```

This fixture deliberately gives torso and pelvis different values so tests prove the implementation selects `pelvis`, not the tracking anchor.

- [ ] **Step 3: Replace the old privileged function test**

Replace `test_teacher_privileged_observation_functions_return_expected_values` with:

```python
def test_teacher_pelvis_observation_functions_return_expected_values() -> None:
  command = _FakeMotionCommand()
  env = _FakeEnv(command)

  assert torch.allclose(
    oracle_obs.reference_pelvis_pos_error_b(env, "motion"),
    torch.full((2, 3), 0.5),
  )
  assert oracle_obs.reference_pelvis_ori_error_b(env, "motion").shape == (2, 6)
  assert torch.allclose(
    oracle_obs.pelvis_lin_vel(env, "motion"),
    torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32),
  )
  assert torch.allclose(
    oracle_obs.pelvis_ang_vel(env, "motion"),
    torch.tensor([[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]], dtype=torch.float32),
  )
  assert torch.allclose(
    oracle_obs.pelvis_global_pos_w(env, "motion"),
    command.robot_body_pos_w[:, 1],
  )
  assert torch.allclose(
    oracle_obs.pelvis_global_lin_vel_w(env, "motion"),
    command.robot_body_lin_vel_w[:, 1],
  )
```

- [ ] **Step 4: Add a missing-pelvis error test**

Append this test after `test_teacher_pelvis_observation_functions_return_expected_values`:

```python
def test_teacher_pelvis_observation_functions_require_pelvis_body() -> None:
  command = _FakeMotionCommand(body_names=("torso_link", "left_hip_roll_link"))
  env = _FakeEnv(command)

  try:
    oracle_obs.pelvis_global_pos_w(env, "motion")
  except ValueError as exc:
    assert "pelvis" in str(exc)
    assert "body_names" in str(exc)
  else:
    raise AssertionError("Expected pelvis_global_pos_w to require a pelvis body")
```

- [ ] **Step 5: Replace the teacher config assertion test**

Replace `test_oracle_teacher_adds_privileged_terms_to_actor_and_critic` with:

```python
def test_oracle_teacher_replaces_anchor_terms_with_pelvis_terms() -> None:
  cfg = unitree_g1_oracle_teacher_terrain_tracking_env_cfg()

  expected_funcs = {
    "reference_pelvis_pos_error_b": oracle_obs.reference_pelvis_pos_error_b,
    "reference_pelvis_ori_error_b": oracle_obs.reference_pelvis_ori_error_b,
    "pelvis_lin_vel": oracle_obs.pelvis_lin_vel,
    "pelvis_ang_vel": oracle_obs.pelvis_ang_vel,
    "pelvis_global_pos_w": oracle_obs.pelvis_global_pos_w,
    "pelvis_global_lin_vel_w": oracle_obs.pelvis_global_lin_vel_w,
  }
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    assert "height_scan" in terms
    assert REPLACED_TEACHER_TERMS.isdisjoint(terms)
    for term_name, func in expected_funcs.items():
      term = terms[term_name]
      assert term.func is func
      assert term.params == {"command_name": "motion"}
      assert term.noise is None
```

- [ ] **Step 6: Run the focused test and verify it fails for the expected reason**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_oracle_terrain_tracking_config.py
```

Expected: FAIL because `oracle_obs.reference_pelvis_pos_error_b` and the other new pelvis observation functions are not implemented yet, or because OracleTeacher still contains the replaced torso/base terms.

- [ ] **Step 7: Commit the failing tests**

```bash
git add tests/test_oracle_terrain_tracking_config.py
git commit -m "test: specify php pelvis teacher observations"
```

## Task 2: Implement Pelvis Observation Functions

**Files:**
- Modify: `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/observations.py`
- Test: `tests/test_oracle_terrain_tracking_config.py`

- [ ] **Step 1: Update imports**

In `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/observations.py`, add the mjlab math imports:

```python
from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms
```

- [ ] **Step 2: Replace old anchor observation helpers with pelvis helpers**

Replace the old `global_anchor_pos_error_w`, `global_anchor_lin_vel_error_w`, and `reference_anchor_lin_vel_w` functions with:

```python
def _body_index(command: MotionCommand, body_name: str) -> int:
  try:
    return command.cfg.body_names.index(body_name)
  except ValueError as exc:
    raise ValueError(
      f"Motion command body_names must include {body_name!r} for PHP teacher observations"
    ) from exc


def _pelvis_index(command: MotionCommand) -> int:
  return _body_index(command, "pelvis")


def reference_pelvis_pos_error_b(env: Any, command_name: str) -> Tensor:
  command = _motion_command(env, command_name)
  pelvis_idx = _pelvis_index(command)
  pos, _ = subtract_frame_transforms(
    command.robot_body_pos_w[:, pelvis_idx],
    command.robot_body_quat_w[:, pelvis_idx],
    command.body_pos_w[:, pelvis_idx],
    command.body_quat_w[:, pelvis_idx],
  )
  return pos.view(env.num_envs, -1)


def reference_pelvis_ori_error_b(env: Any, command_name: str) -> Tensor:
  command = _motion_command(env, command_name)
  pelvis_idx = _pelvis_index(command)
  _, ori = subtract_frame_transforms(
    command.robot_body_pos_w[:, pelvis_idx],
    command.robot_body_quat_w[:, pelvis_idx],
    command.body_pos_w[:, pelvis_idx],
    command.body_quat_w[:, pelvis_idx],
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def pelvis_lin_vel(env: Any, command_name: str) -> Tensor:
  command = _motion_command(env, command_name)
  pelvis_idx = _pelvis_index(command)
  return command.robot_body_lin_vel_w[:, pelvis_idx]


def pelvis_ang_vel(env: Any, command_name: str) -> Tensor:
  command = _motion_command(env, command_name)
  pelvis_idx = _pelvis_index(command)
  return command.robot_body_ang_vel_w[:, pelvis_idx]


def pelvis_global_pos_w(env: Any, command_name: str) -> Tensor:
  command = _motion_command(env, command_name)
  pelvis_idx = _pelvis_index(command)
  return command.robot_body_pos_w[:, pelvis_idx]


def pelvis_global_lin_vel_w(env: Any, command_name: str) -> Tensor:
  command = _motion_command(env, command_name)
  pelvis_idx = _pelvis_index(command)
  return command.robot_body_lin_vel_w[:, pelvis_idx]
```

- [ ] **Step 3: Run the focused function tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_oracle_terrain_tracking_config.py::test_teacher_pelvis_observation_functions_return_expected_values tests/test_oracle_terrain_tracking_config.py::test_teacher_pelvis_observation_functions_require_pelvis_body
```

Expected: PASS. If orientation shape fails, inspect `matrix_from_quat` output and keep the same two-column encoding used by mjlab's `motion_anchor_ori_b`.

- [ ] **Step 4: Commit the observation functions**

```bash
git add src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/observations.py tests/test_oracle_terrain_tracking_config.py
git commit -m "feat: add pelvis teacher observation functions"
```

## Task 3: Wire Pelvis Observations Into OracleTeacher

**Files:**
- Modify: `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/env_cfgs.py`
- Test: `tests/test_oracle_terrain_tracking_config.py`

- [ ] **Step 1: Replace `_add_oracle_teacher_terms`**

In `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/env_cfgs.py`, replace `_add_oracle_teacher_terms` with:

```python
def _add_oracle_teacher_terms(cfg: ManagerBasedRlEnvCfg) -> None:
  teacher_terms = {
    "reference_pelvis_pos_error_b": observations.reference_pelvis_pos_error_b,
    "reference_pelvis_ori_error_b": observations.reference_pelvis_ori_error_b,
    "pelvis_lin_vel": observations.pelvis_lin_vel,
    "pelvis_ang_vel": observations.pelvis_ang_vel,
    "pelvis_global_pos_w": observations.pelvis_global_pos_w,
    "pelvis_global_lin_vel_w": observations.pelvis_global_lin_vel_w,
  }
  replaced_terms = {
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "base_lin_vel",
    "base_ang_vel",
    "global_anchor_pos_error_w",
    "global_anchor_lin_vel_error_w",
    "reference_anchor_lin_vel_w",
  }
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    for term_name in replaced_terms:
      terms.pop(term_name, None)
    for term_name, func in teacher_terms.items():
      terms[term_name] = _teacher_term(func)
```

This removes the inherited torso-anchor pose terms and base velocity terms only from OracleTeacher. OracleHeight remains unchanged because `_add_oracle_teacher_terms` is not called for it.

- [ ] **Step 2: Run the focused config tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_oracle_terrain_tracking_config.py
```

Expected: PASS.

- [ ] **Step 3: Run the oracle smoke tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_env_smoke.py -k oracle_terrain
```

Expected: PASS. If the expected actor observation dimension in `tests/test_env_smoke.py` fails, compute the new dimension difference. The OracleTeacher dimension should change because it removes `motion_anchor_pos_b` (3), `motion_anchor_ori_b` (6), `base_lin_vel` (3), `base_ang_vel` (3), and old privileged terms (9), then adds pelvis pose error (3 + 6), pelvis velocities (3 + 3), and pelvis privileged terms (3 + 3). Net expected actor dimension change from the previous OracleTeacher is `-9`.

- [ ] **Step 4: Update smoke-test dimension expectations if needed**

Only if Step 3 fails on `min_extra_actor_dims`, update the OracleTeacher expected lower bound in `tests/test_env_smoke.py` from `73` to `64`.

Patch:

```python
@pytest.mark.parametrize(
  ("task_id", "min_extra_actor_dims"),
  [
    ("TT-Tracking-TerrainOracleHeight-Unitree-G1", 64),
    ("TT-Tracking-TerrainOracleTeacher-Unitree-G1", 64),
  ],
)
```

Then rerun:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_env_smoke.py -k oracle_terrain
```

Expected: PASS.

- [ ] **Step 5: Commit the OracleTeacher wiring**

```bash
git add src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/env_cfgs.py tests/test_env_smoke.py
git commit -m "feat: align oracle teacher observations with php"
```

## Task 4: Final Verification

**Files:**
- Verify: `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/observations.py`
- Verify: `src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/env_cfgs.py`
- Verify: `tests/test_oracle_terrain_tracking_config.py`
- Verify: `tests/test_env_smoke.py`

- [ ] **Step 1: Run focused oracle tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest python -m pytest -q tests/test_oracle_terrain_tracking_config.py tests/test_env_smoke.py -k "oracle_terrain or oracle_teacher or oracle_height"
```

Expected: PASS.

- [ ] **Step 2: Inspect the staged diff for unrelated changes**

Run:

```bash
git status --short
git diff -- src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/observations.py src/terrain_tracking/tasks/oracle_terrain_tracking/config/g1/env_cfgs.py tests/test_oracle_terrain_tracking_config.py tests/test_env_smoke.py
```

Expected: only OracleTeacher observation alignment changes appear. Existing unrelated workspace changes in scripts and untracked files may still appear in `git status`; do not revert or commit them.

- [ ] **Step 3: Commit any final test-only adjustment if Step 4 in Task 3 was needed**

If `tests/test_env_smoke.py` was changed after the previous commit, run:

```bash
git add tests/test_env_smoke.py
git commit -m "test: update oracle teacher smoke dimension"
```

Expected: commit succeeds. Skip this step if there are no uncommitted changes in `tests/test_env_smoke.py`.

## Self-Review

- Spec coverage: The plan covers pelvis-specific basic observations, PHP-style privileged pelvis global observations, keeping `torso_link` as tracking anchor, preserving torso-mounted height scan, applying terms to actor and critic, and protecting OracleHeight from teacher terms.
- Red-flag scan: The plan contains no unresolved markers and gives exact file paths, code snippets, commands, and expected outcomes.
- Type consistency: Function names match the approved spec and are used consistently across tests, implementation, and env wiring.
