# General Tracker Sampling and Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Stabilize the general pair-dataset tracker by keeping pair-level coverage static, making frame-level adaptive sampling slow and floor-regularized, and exposing truthful sampler/tracking metrics.

**Architecture:** Keep general tracker reset sampling as a two-level process: sample a pair from static record weights, then sample a local frame inside that pair. Only the per-pair frame-bin distribution adapts from failures. `MultiMotionCommand` owns training semantics and metric reporting; `PairFrameSampler` owns probability state and sampling math.

**Tech Stack:** Python 3.13 project runtime, torch, mjlab command terms, pytest, TensorBoard scalar logging through command metrics.

---

## File Map

- Modify `src/terrain_tracking/runtime/pair_frame_sampler.py`: replace aggressive EMA-to-probability updates with slow failure-score adaptive sampling, add probability/entropy helpers, preserve valid independent sampling.
- Modify `src/terrain_tracking/tasks/general_terrain_tracking/mdp/multi_motion_command.py`: fix `start` pair sampling, wire sampler failure updates, add truthful sampler metrics and tracking error metrics.
- Modify `src/terrain_tracking/tasks/general_terrain_tracking/scripts/train.py`: copy new sampler cfg fields from task defaults into CLI frontend overrides.
- Modify `src/terrain_tracking/tasks/general_terrain_tracking/scripts/play.py`: same frontend cfg copy for play/eval consistency.
- Modify `tests/test_pair_frame_sampler.py`: cover slow adaptive behavior, floor regularization, pair coverage, probability helper semantics.
- Modify `tests/test_multi_motion_command_unit.py`: assert cfg exposes new sampler fields and no longer depends on `ema_alpha`.
- Modify `tests/test_general_env_smoke.py`: assert `start` mode can sample multiple pairs and command metrics include real sampling/tracking keys.

## Design Decisions

- Pair selection remains static by default: `pair_probs = normalize(record.weight)`.
- Frame-bin selection adapts per pair: `bin_probs[pair] = normalize(bin_failure_score[pair] + adaptive_uniform_ratio / num_bins)`.
- Failure scores update slowly: `score = (1 - adaptive_alpha) * score + adaptive_alpha * count`.
- Default values follow single `MotionCommand` scale: `adaptive_alpha=0.001`, `adaptive_uniform_ratio=0.1`.
- `sampling_mode=start` means "sample a pair, start at frame 0"; it must not force `pair_indices=0`.
- Metrics distinguish pair distribution, conditional bin distribution, joint pair-bin distribution, active env occupancy, and tracking error.
- `concat_pair_bins` remains available as an experimental mode but is not the default and must also use slow/floor-regularized probabilities if exercised.

## Task 1: Replace Aggressive PairFrameSampler Updates

**Files:**
- Modify: `src/terrain_tracking/runtime/pair_frame_sampler.py`
- Test: `tests/test_pair_frame_sampler.py`

- [x] **Step 1: Add failing sampler tests**

Append these tests to `tests/test_pair_frame_sampler.py`:

```python
def test_pair_frame_sampler_adaptive_keeps_probability_floor() -> None:
  sampler = PairFrameSampler(
    frame_counts=torch.tensor([100]),
    cfg=PairFrameSamplerCfg(
      num_bins=10,
      mode="independent",
      adaptive_alpha=0.001,
      adaptive_uniform_ratio=0.1,
      min_weight=0.0,
    ),
    device="cpu",
  )

  pair_indices = torch.zeros(1000, dtype=torch.long)
  local_frames = torch.zeros(1000, dtype=torch.long)
  failure_mask = torch.ones(1000, dtype=torch.bool)
  for _ in range(100):
    sampler.update_failures(pair_indices, local_frames, failure_mask)

  top1 = float(sampler.bin_weights[0].max())
  assert top1 < 0.55
  assert float(sampler.bin_weights[0].min()) > 0.0
  torch.testing.assert_close(sampler.bin_weights.sum(dim=1), torch.ones(1))


def test_pair_frame_sampler_pair_distribution_stays_record_weighted() -> None:
  sampler = PairFrameSampler(
    frame_counts=torch.tensor([20, 20, 20]),
    cfg=PairFrameSamplerCfg(
      num_bins=5,
      mode="independent",
      adaptive_alpha=0.001,
      adaptive_uniform_ratio=0.1,
    ),
    device="cpu",
    pair_weights=torch.tensor([1.0, 2.0, 1.0]),
  )

  pair_indices = torch.zeros(500, dtype=torch.long)
  local_frames = torch.zeros(500, dtype=torch.long)
  failure_mask = torch.ones(500, dtype=torch.bool)
  for _ in range(50):
    sampler.update_failures(pair_indices, local_frames, failure_mask)

  torch.testing.assert_close(
    sampler.pair_probabilities,
    torch.tensor([0.25, 0.5, 0.25]),
  )


def test_pair_frame_sampler_joint_probabilities_match_pair_times_bin_probs() -> None:
  sampler = PairFrameSampler(
    frame_counts=torch.tensor([10, 10]),
    cfg=PairFrameSamplerCfg(num_bins=4, mode="independent"),
    device="cpu",
    pair_weights=torch.tensor([1.0, 3.0]),
  )

  joint = sampler.joint_bin_probabilities

  assert joint.shape == (2, 4)
  torch.testing.assert_close(joint.sum(), torch.tensor(1.0))
  torch.testing.assert_close(joint.sum(dim=1), sampler.pair_probabilities)
```

Update the existing duplicate-bin test to assert slow probability movement instead of one-hot replacement:

```python
def test_pair_frame_sampler_accumulates_duplicate_failure_bins():
  sampler = PairFrameSampler(
    frame_counts=torch.tensor([10]),
    cfg=PairFrameSamplerCfg(
      num_bins=5,
      mode="independent",
      adaptive_alpha=1.0,
      adaptive_uniform_ratio=0.0,
      min_weight=0.0,
    ),
    device="cpu",
  )

  sampler.update_failures(
    pair_indices=torch.zeros(5, dtype=torch.long),
    local_frames=torch.tensor([4, 4, 4, 4, 0]),
    failure_mask=torch.ones(5, dtype=torch.bool),
  )

  torch.testing.assert_close(
    sampler.bin_weights[0],
    torch.tensor([0.2, 0.0, 0.8, 0.0, 0.0]),
  )
```

- [x] **Step 2: Run tests and verify red**

Run:

```bash
uv run pytest -q tests/test_pair_frame_sampler.py
```

Expected: fails because `PairFrameSamplerCfg` has no `adaptive_alpha` or `adaptive_uniform_ratio`, and `PairFrameSampler` has no `pair_probabilities` or `joint_bin_probabilities`.

- [x] **Step 3: Implement failure-score sampler state**

In `src/terrain_tracking/runtime/pair_frame_sampler.py`, change `PairFrameSamplerCfg` to:

```python
@dataclass(frozen=True)
class PairFrameSamplerCfg:
  num_bins: int = 32
  mode: SamplerMode = "independent"
  adaptive_alpha: float = 1.0e-3
  adaptive_uniform_ratio: float = 0.1
  min_weight: float = 1.0e-4
```

In `PairFrameSampler.__init__`, replace direct mutable probability state with score state plus derived probabilities:

```python
self.bin_failure_scores = torch.zeros(
  (self.num_pairs, cfg.num_bins),
  dtype=torch.float32,
  device=self.device,
)
self.bin_weights = self._bin_probabilities_from_scores()
self.concat_bin_weights = self.joint_bin_probabilities.reshape(-1)
```

Add these properties/helpers:

```python
@property
def pair_probabilities(self) -> torch.Tensor:
  return self.pair_weights

@property
def joint_bin_probabilities(self) -> torch.Tensor:
  return self.pair_weights[:, None] * self.bin_weights

def _bin_probabilities_from_scores(self) -> torch.Tensor:
  floor = self.cfg.adaptive_uniform_ratio / float(self.cfg.num_bins)
  return _normalize_rows(
    self.bin_failure_scores + floor,
    min_weight=self.cfg.min_weight,
  )

def _refresh_probabilities(self) -> None:
  self.bin_weights = self._bin_probabilities_from_scores()
  self.concat_bin_weights = self.joint_bin_probabilities.reshape(-1)
```

Add row-wise normalization:

```python
def _normalize_rows(weights: torch.Tensor, *, min_weight: float) -> torch.Tensor:
  weights = torch.clamp(weights, min=min_weight)
  return weights / torch.sum(weights, dim=1, keepdim=True)
```

Keep `_normalize()` for 1D pair weights.

- [x] **Step 4: Update failure scoring**

Replace the current `update_failures()` probability blend with:

```python
counts = torch.zeros_like(self.bin_failure_scores)
counts.index_put_(
  (failed_pairs, failed_bins),
  torch.ones_like(failed_bins, dtype=counts.dtype),
  accumulate=True,
)
self.bin_failure_scores = (
  (1.0 - self.cfg.adaptive_alpha) * self.bin_failure_scores
  + self.cfg.adaptive_alpha * counts
)
self._refresh_probabilities()
```

This intentionally updates every row with EMA decay. Pairs without current failures decay slowly toward uniform-floor behavior.

- [x] **Step 5: Run sampler tests**

Run:

```bash
uv run pytest -q tests/test_pair_frame_sampler.py
```

Expected: pass.

## Task 2: Fix MultiMotionCommand Sampling Modes

**Files:**
- Modify: `src/terrain_tracking/tasks/general_terrain_tracking/mdp/multi_motion_command.py`
- Test: `tests/test_general_env_smoke.py`

- [x] **Step 1: Add failing start-mode coverage test**

Append to `tests/test_general_env_smoke.py`:

```python
def test_general_start_sampling_uses_multiple_pairs(tmp_path: Path) -> None:
  root = tmp_path / "parc"
  _create_pair(root, "platform/a", frames=8)
  _create_pair(root, "stairs/b", frames=9)
  manifest = tmp_path / "pair_dataset.jsonl"
  build_parc_pair_dataset(BuildPairDatasetConfig(root=root, output=manifest))

  cfg = unitree_g1_general_oracle_teacher_terrain_tracking_env_cfg(
    pair_dataset=str(manifest),
    play=True,
  )
  cfg.scene.num_envs = 16
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MultiMotionCommandCfg)
  motion_cmd.sampling_mode = "start"

  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset()
    command = env.command_manager.get_term("motion")
    assert torch.all(command.time_steps == 0)
    assert torch.unique(command.env_pair_indices).numel() > 1
  finally:
    env.close()
```

- [x] **Step 2: Run test and verify red**

Run:

```bash
uv run pytest -q tests/test_general_env_smoke.py::test_general_start_sampling_uses_multiple_pairs
```

Expected: fails because `sampling_mode=start` currently sets all `pair_indices` to zero.

- [x] **Step 3: Fix `_resample_command()` pair selection**

In `MultiMotionCommand._resample_command()`, replace the `start` branch with:

```python
if self.cfg.sampling_mode == "start":
  sample = self.sampler.sample(env_ids)
  pair_indices = sample.pair_indices
  local_frames = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
else:
  if self.cfg.sampling_mode == "adaptive" and hasattr(
    self._env,
    "termination_manager",
  ):
    self.sampler.update_failures(
      self.env_pair_indices[env_ids],
      self.time_steps[env_ids],
      self._env.termination_manager.terminated[env_ids],
    )
  sample = self.sampler.sample(env_ids)
  pair_indices = sample.pair_indices
  local_frames = sample.local_frames
```

Then add an explicit uniform branch if desired for readability:

```python
elif self.cfg.sampling_mode == "uniform":
  sample = self.sampler.sample(env_ids)
  pair_indices = sample.pair_indices
  local_frames = sample.local_frames
else:
  assert self.cfg.sampling_mode == "adaptive"
  ...
```

The important invariant is: `start` samples pair normally and forces frame 0.

- [x] **Step 4: Run start-mode test**

Run:

```bash
uv run pytest -q tests/test_general_env_smoke.py::test_general_start_sampling_uses_multiple_pairs
```

Expected: pass.

## Task 3: Update CLI Frontend Sampler Copying

**Files:**
- Modify: `src/terrain_tracking/tasks/general_terrain_tracking/scripts/train.py`
- Modify: `src/terrain_tracking/tasks/general_terrain_tracking/scripts/play.py`
- Test: `tests/test_cli_help.py` and `tests/test_multi_motion_command_unit.py`

- [x] **Step 1: Update cfg field assertions**

In `tests/test_multi_motion_command_unit.py`, extend `test_multi_motion_command_cfg_exposes_general_dataset_fields()`:

```python
assert cfg.sampler.adaptive_alpha == 1.0e-3
assert cfg.sampler.adaptive_uniform_ratio == 0.1
assert not hasattr(cfg.sampler, "ema_alpha")
```

- [x] **Step 2: Run and verify red**

Run:

```bash
uv run pytest -q tests/test_multi_motion_command_unit.py
```

Expected: fails until cfg fields are updated and references to `ema_alpha` are removed.

- [x] **Step 3: Update train frontend**

In `src/terrain_tracking/tasks/general_terrain_tracking/scripts/train.py`, change the sampler copy block to:

```python
motion_cmd.sampler = PairFrameSamplerCfg(
  num_bins=motion_cmd.sampler.num_bins,
  mode=frontend.pair_sampler_mode,
  adaptive_alpha=motion_cmd.sampler.adaptive_alpha,
  adaptive_uniform_ratio=motion_cmd.sampler.adaptive_uniform_ratio,
  min_weight=motion_cmd.sampler.min_weight,
)
```

- [x] **Step 4: Update play frontend**

In `src/terrain_tracking/tasks/general_terrain_tracking/scripts/play.py`, make the same replacement:

```python
motion_cmd.sampler = PairFrameSamplerCfg(
  num_bins=motion_cmd.sampler.num_bins,
  mode=args.pair_sampler_mode,
  adaptive_alpha=motion_cmd.sampler.adaptive_alpha,
  adaptive_uniform_ratio=motion_cmd.sampler.adaptive_uniform_ratio,
  min_weight=motion_cmd.sampler.min_weight,
)
```

- [x] **Step 5: Run targeted tests**

Run:

```bash
uv run pytest -q tests/test_multi_motion_command_unit.py tests/test_cli_help.py
```

Expected: pass.

## Task 4: Add Truthful Sampling Metrics

**Files:**
- Modify: `src/terrain_tracking/tasks/general_terrain_tracking/mdp/multi_motion_command.py`
- Test: `tests/test_general_env_smoke.py`

- [x] **Step 1: Add smoke assertions for metric keys**

In `test_general_oracle_teacher_env_can_reset_and_step_on_cpu()`, after `command = env.command_manager.get_term("motion")`, add:

```python
expected_metric_keys = {
  "sampling_entropy",
  "sampling_top1_prob",
  "sampling_pair_entropy",
  "sampling_pair_top1_prob",
  "sampling_bin_entropy_mean",
  "sampling_bin_entropy_min",
  "sampling_bin_top1_prob_mean",
  "sampling_bin_top1_prob_max",
  "active_pair_entropy",
  "active_pair_top1_frac",
}
assert expected_metric_keys.issubset(command.metrics)
```

After one step, add:

```python
for key in expected_metric_keys:
  value = command.metrics[key]
  assert value.shape == (env.num_envs,)
  assert torch.isfinite(value).all()
```

- [x] **Step 2: Run test and verify red**

Run:

```bash
uv run pytest -q tests/test_general_env_smoke.py::test_general_oracle_teacher_env_can_reset_and_step_on_cpu
```

Expected: fails because metrics are missing.

- [x] **Step 3: Initialize metric tensors**

In `MultiMotionCommand.__init__`, replace the two sampling metric initializers with:

```python
for metric_name in (
  "sampling_entropy",
  "sampling_top1_prob",
  "sampling_pair_entropy",
  "sampling_pair_top1_prob",
  "sampling_bin_entropy_mean",
  "sampling_bin_entropy_min",
  "sampling_bin_top1_prob_mean",
  "sampling_bin_top1_prob_max",
  "active_pair_entropy",
  "active_pair_top1_frac",
):
  self.metrics[metric_name] = torch.zeros(self.num_envs, device=self.device)
```

- [x] **Step 4: Add entropy helper**

Near `MultiMotionCommand._update_metrics()`, add:

```python
def _normalized_entropy(probabilities: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
  count = probabilities.shape[dim]
  entropy = -torch.sum(
    probabilities * torch.log(probabilities + 1.0e-12),
    dim=dim,
  )
  if count <= 1:
    return torch.ones_like(entropy)
  return entropy / torch.log(
    torch.tensor(float(count), device=probabilities.device, dtype=probabilities.dtype)
  )
```

- [x] **Step 5: Compute sampling metrics in `_update_metrics()`**

At the top of `_update_metrics()`, compute:

```python
pair_probs = self.sampler.pair_probabilities
bin_probs = self.sampler.bin_weights
joint_probs = self.sampler.joint_bin_probabilities.reshape(-1)

pair_entropy = _normalized_entropy(pair_probs)
pair_top1 = torch.max(pair_probs)
bin_entropy = _normalized_entropy(bin_probs, dim=1)
bin_top1 = torch.max(bin_probs, dim=1).values
joint_entropy = _normalized_entropy(joint_probs)
joint_top1 = torch.max(joint_probs)

active_counts = torch.bincount(
  self.env_pair_indices,
  minlength=self.motion.frame_counts.numel(),
).to(dtype=torch.float32)
active_probs = active_counts / torch.clamp(active_counts.sum(), min=1.0)
active_entropy = _normalized_entropy(active_probs)
active_top1 = torch.max(active_probs)

self.metrics["sampling_entropy"][:] = joint_entropy
self.metrics["sampling_top1_prob"][:] = joint_top1
self.metrics["sampling_pair_entropy"][:] = pair_entropy
self.metrics["sampling_pair_top1_prob"][:] = pair_top1
self.metrics["sampling_bin_entropy_mean"][:] = torch.mean(bin_entropy)
self.metrics["sampling_bin_entropy_min"][:] = torch.min(bin_entropy)
self.metrics["sampling_bin_top1_prob_mean"][:] = torch.mean(bin_top1)
self.metrics["sampling_bin_top1_prob_max"][:] = torch.max(bin_top1)
self.metrics["active_pair_entropy"][:] = active_entropy
self.metrics["active_pair_top1_frac"][:] = active_top1
```

- [x] **Step 6: Run smoke test**

Run:

```bash
uv run pytest -q tests/test_general_env_smoke.py::test_general_oracle_teacher_env_can_reset_and_step_on_cpu
```

Expected: pass.

## Task 5: Add Tracking Error Metrics

**Files:**
- Modify: `src/terrain_tracking/tasks/general_terrain_tracking/mdp/multi_motion_command.py`
- Test: `tests/test_general_env_smoke.py`

- [x] **Step 1: Add tracking metric assertions**

In `test_general_oracle_teacher_env_can_reset_and_step_on_cpu()`, extend `expected_metric_keys` with:

```python
tracking_metric_keys = {
  "error_anchor_pos",
  "error_anchor_rot",
  "error_anchor_lin_vel",
  "error_anchor_ang_vel",
  "error_body_pos",
  "error_body_rot",
  "error_body_lin_vel",
  "error_body_ang_vel",
  "error_joint_pos",
  "error_joint_vel",
}
assert tracking_metric_keys.issubset(command.metrics)
```

After one step, assert finite values:

```python
for key in tracking_metric_keys:
  value = command.metrics[key]
  assert value.shape == (env.num_envs,)
  assert torch.isfinite(value).all()
```

- [x] **Step 2: Run test and verify red**

Run:

```bash
uv run pytest -q tests/test_general_env_smoke.py::test_general_oracle_teacher_env_can_reset_and_step_on_cpu
```

Expected: fails because tracking metrics are missing.

- [x] **Step 3: Import quat error helper**

In `src/terrain_tracking/tasks/general_terrain_tracking/mdp/multi_motion_command.py`, add `quat_error_magnitude` to the math imports:

```python
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_error_magnitude,
  quat_inv,
  quat_mul,
  yaw_quat,
)
```

- [x] **Step 4: Initialize tracking metric tensors**

In `MultiMotionCommand.__init__`, add:

```python
for metric_name in (
  "error_anchor_pos",
  "error_anchor_rot",
  "error_anchor_lin_vel",
  "error_anchor_ang_vel",
  "error_body_pos",
  "error_body_rot",
  "error_body_lin_vel",
  "error_body_ang_vel",
  "error_joint_pos",
  "error_joint_vel",
):
  self.metrics[metric_name] = torch.zeros(self.num_envs, device=self.device)
```

- [x] **Step 5: Compute tracking metrics**

At the end of `_update_metrics()`, after sampling metric assignments, add:

```python
self.metrics["error_anchor_pos"] = torch.norm(
  self.anchor_pos_w - self.robot_anchor_pos_w,
  dim=-1,
)
self.metrics["error_anchor_rot"] = quat_error_magnitude(
  self.anchor_quat_w,
  self.robot_anchor_quat_w,
)
self.metrics["error_anchor_lin_vel"] = torch.norm(
  self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w,
  dim=-1,
)
self.metrics["error_anchor_ang_vel"] = torch.norm(
  self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w,
  dim=-1,
)
self.metrics["error_body_pos"] = torch.norm(
  self.body_pos_relative_w - self.robot_body_pos_w,
  dim=-1,
).mean(dim=-1)
self.metrics["error_body_rot"] = quat_error_magnitude(
  self.body_quat_relative_w,
  self.robot_body_quat_w,
).mean(dim=-1)
self.metrics["error_body_lin_vel"] = torch.norm(
  self.body_lin_vel_w - self.robot_body_lin_vel_w,
  dim=-1,
).mean(dim=-1)
self.metrics["error_body_ang_vel"] = torch.norm(
  self.body_ang_vel_w - self.robot_body_ang_vel_w,
  dim=-1,
).mean(dim=-1)
self.metrics["error_joint_pos"] = torch.norm(
  self.joint_pos - self.robot_joint_pos,
  dim=-1,
)
self.metrics["error_joint_vel"] = torch.norm(
  self.joint_vel - self.robot_joint_vel,
  dim=-1,
)
```

- [x] **Step 6: Run smoke test**

Run:

```bash
uv run pytest -q tests/test_general_env_smoke.py::test_general_oracle_teacher_env_can_reset_and_step_on_cpu
```

Expected: pass.

## Task 6: Full Verification and Canary Guidance

**Files:**
- No code files.
- Uses the modified tests and a short training canary.

- [x] **Step 1: Run focused test suite**

Run:

```bash
uv run pytest -q \
  tests/test_pair_frame_sampler.py \
  tests/test_multi_motion_command_unit.py \
  tests/test_general_terrain_tracking_config.py \
  tests/test_general_env_smoke.py \
  tests/test_cli_help.py
```

Expected: pass.

- [x] **Step 2: Run broader related tests**

Run:

```bash
uv run pytest -q \
  tests/test_pair_dataset.py \
  tests/test_motion_library.py \
  tests/test_pair_terrain_bank.py \
  tests/test_out_of_tile_bounds.py
```

Expected: pass.

- [ ] **Step 3: Run a short general-tracker canary (deferred)**

Run a small canary after the tile-local observation fix lands:

```bash
NUM_ENVS=1024 MAX_ITERATIONS=300 \
scripts/exp/train/mid_blocks_general_oracle_teacher_adaptive.sh
```

Expected TensorBoard behavior:

- `Metrics/motion/sampling_pair_top1_prob` near `1 / number_of_pairs` for equal weights.
- `Metrics/motion/sampling_bin_top1_prob_max` does not jump to `0.9969` in the first few hundred iterations.
- `Metrics/motion/sampling_entropy` is not hard-coded to `1.0`; it reflects joint pair-bin distribution.
- `Metrics/motion/error_anchor_pos`, `error_body_pos`, and `error_joint_vel` are logged and should trend down if the policy learns.
- `Train/mean_episode_length` should improve beyond the previous failed general run's early `40-90` range before spending a full training budget.

## Self-Review

- Spec coverage: covers two-level sampling semantics, slow frame-level adaptive sampling, start-mode pair coverage, truthful sampler metrics, and tracking metrics. Tile-local observation is intentionally excluded because it is assigned to another agent.
- Placeholder scan: no placeholder tasks or unspecified implementation points remain.
- Type consistency: `PairFrameSamplerCfg` consistently uses `adaptive_alpha`, `adaptive_uniform_ratio`, and `min_weight`; frontend scripts copy those fields; tests no longer rely on `ema_alpha`.
