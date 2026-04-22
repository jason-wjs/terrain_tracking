from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

from terrain_tracking.runtime.pair_manifest import PairManifest
from terrain_tracking.scene.paired_mesh_spec import make_paired_mesh_spec_fn


def apply_pair_manifest_to_env_cfg(
  cfg: ManagerBasedRlEnvCfg,
  manifest: str | Path | PairManifest,
) -> PairManifest:
  pair = manifest if isinstance(manifest, PairManifest) else PairManifest.load(manifest)

  motion_cfg = cfg.commands["motion"]
  if not isinstance(motion_cfg, MotionCommandCfg):
    raise TypeError("Expected cfg.commands['motion'] to be a MotionCommandCfg")

  motion_cfg.motion_file = str(pair.motion_file)

  existing_spec_fn = cfg.scene.spec_fn
  pair_spec_fn = make_paired_mesh_spec_fn(
    pair,
    num_envs=cfg.scene.num_envs,
    env_spacing=cfg.scene.env_spacing,
  )

  if existing_spec_fn is None:
    cfg.scene.spec_fn = pair_spec_fn
  else:

    def chained_spec_fn(spec) -> None:
      existing_spec_fn(spec)
      pair_spec_fn(spec)

    cfg.scene.spec_fn = chained_spec_fn

  return pair
