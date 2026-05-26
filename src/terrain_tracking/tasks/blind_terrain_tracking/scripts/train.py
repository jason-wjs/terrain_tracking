from __future__ import annotations

import sys
from dataclasses import dataclass

import mjlab
import tyro
from mjlab.scripts.train import TrainConfig as MjlabTrainConfig
from mjlab.scripts.train import launch_training

from terrain_tracking.runtime.apply_pair import apply_pair_manifest_to_env_cfg
from terrain_tracking.tasks.blind_terrain_tracking.scripts.common import (
  DEFAULT_TASK_ID,
  import_local_tasks,
)


@dataclass(frozen=True)
class FrontendConfig:
  pair_manifest: str
  task: str = DEFAULT_TASK_ID
  collision_backend: str = "primitive_boxes"


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
  apply_pair_manifest_to_env_cfg(
    args.env,
    frontend.pair_manifest,
    collision_backend=frontend.collision_backend,
  )
  launch_training(task_id=frontend.task, args=args)


if __name__ == "__main__":
  main()
