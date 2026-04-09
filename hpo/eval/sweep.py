"""Launch eval jobs for all checkpoints belonging to an SFT sweep.

Discovers every <ckpt_root>/<run_id>/step_* directory whose config.yaml
contains the requested sweep_id, then calls eval/launch.py --multirun
with all matching ckpt values so the Hydra launcher submits one job per
checkpoint.

Usage:
    python hpo/eval/sweep.py sweep_id=sweep_0
    python hpo/eval/sweep.py sweep_id=sweep_0 eval_suite=capability safety_suite=safety_core
    python hpo/eval/sweep.py sweep_id=sweep_0 ckpt_root=/other/results/root
"""

import os
import subprocess
import sys

import hydra
import yaml
from omegaconf import MISSING, OmegaConf
from dataclasses import dataclass


@dataclass
class SweepLaunchConfig:
    sweep_id: str = MISSING
    ckpt_root: str = "/home/l/leog/links/projects/aip-glaj/leog/results"


from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(name="sweep_schema", node=SweepLaunchConfig)


@hydra.main(version_base=None, config_path="configs", config_name="sweep")
def main(cfg: SweepLaunchConfig) -> None:
    repo_root = hydra.utils.get_original_cwd()

    # Discover all checkpoints for the given sweep_id.
    ckpts = []
    ckpt_root = cfg.ckpt_root
    for run_id in sorted(os.listdir(ckpt_root)):
        run_dir = os.path.join(ckpt_root, run_id)
        config_path = os.path.join(run_dir, "config.yaml")
        if not os.path.isfile(config_path):
            continue
        with open(config_path) as f:
            run_cfg = yaml.safe_load(f)
        if run_cfg.get("sweep_id") != cfg.sweep_id:
            continue
        for entry in sorted(os.listdir(run_dir)):
            if entry.startswith(("step_", "epoch_")) and os.path.isdir(os.path.join(run_dir, entry)):
                ckpts.append(f"{run_id}/{entry}")

    if not ckpts:
        print(f"No checkpoints found for sweep_id={cfg.sweep_id!r} under {ckpt_root}")
        sys.exit(1)

    print(f"Found {len(ckpts)} checkpoints for sweep_id={cfg.sweep_id!r}:")
    for c in ckpts:
        print(f"  {c}")

    # Forward any extra overrides passed to this script (e.g. eval_suite=, safety_suite=)
    # by re-reading sys.argv, stripping the sweep-specific ones.
    sweep_keys = {"sweep_id", "ckpt_root"}
    extra_overrides = [
        arg for arg in sys.argv[1:]
        if not any(arg.startswith(f"{k}=") for k in sweep_keys)
        and not arg.startswith("--")
    ]

    ckpt_sweep = ",".join(ckpts)
    cmd = [
        "python", os.path.join(repo_root, "hpo/eval/launch.py"),
        "--multirun",
        f"ckpt={ckpt_sweep}",
        f"ckpt_root={ckpt_root}",
    ] + extra_overrides

    print("\nLaunching:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
