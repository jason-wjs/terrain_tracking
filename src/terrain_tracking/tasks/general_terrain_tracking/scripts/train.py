from __future__ import annotations

import sys
from dataclasses import dataclass

import mjlab
import tyro
from mjlab.scripts.train import TrainConfig as MjlabTrainConfig
from mjlab.scripts.train import launch_training

from terrain_tracking.runtime.apply_pair_dataset import apply_pair_dataset_to_env_cfg
from terrain_tracking.runtime.pair_dataset import ValidationMode
from terrain_tracking.runtime.pair_frame_sampler import PairFrameSamplerCfg, SamplerMode
from terrain_tracking.tasks.blind_terrain_tracking.scripts.common import (
  import_local_tasks,
)
from terrain_tracking.tasks.general_terrain_tracking.mdp import MultiMotionCommandCfg

DEFAULT_TASK_ID = "TT-Tracking-TerrainOracleTeacherGeneral-Unitree-G1"


@dataclass(frozen=True)
class FrontendConfig:
  pair_dataset: str
  task: str = DEFAULT_TASK_ID
  dataset_validate: ValidationMode = "fast"
  max_pairs: int | None = None
  pair_filter: tuple[str, ...] = ()
  pair_sampler_mode: SamplerMode = "independent"


def _apply_frontend(args: MjlabTrainConfig, frontend: FrontendConfig) -> None:
  motion_cmd = args.env.commands["motion"]
  if not isinstance(motion_cmd, MultiMotionCommandCfg):
    raise TypeError("Expected cfg.commands['motion'] to be a MultiMotionCommandCfg")
  motion_cmd.pair_dataset = frontend.pair_dataset
  motion_cmd.dataset_validate = frontend.dataset_validate
  motion_cmd.max_pairs = frontend.max_pairs
  motion_cmd.pair_filter = frontend.pair_filter
  motion_cmd.sampler = PairFrameSamplerCfg(
    num_bins=motion_cmd.sampler.num_bins,
    mode=frontend.pair_sampler_mode,
    ema_alpha=motion_cmd.sampler.ema_alpha,
    min_weight=motion_cmd.sampler.min_weight,
  )
  apply_pair_dataset_to_env_cfg(args.env)


def main() -> None:
  import_local_tasks()
  frontend, remaining_args = tyro.cli(
    FrontendConfig,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  train_cfg = MjlabTrainConfig.from_task(frontend.task)
  args = tyro.cli(
    MjlabTrainConfig,
    args=remaining_args,
    default=train_cfg,
    prog=sys.argv[0],
    config=mjlab.TYRO_FLAGS,
  )
  _apply_frontend(args, frontend)
  launch_training(task_id=frontend.task, args=args)


if __name__ == "__main__":
  main()
