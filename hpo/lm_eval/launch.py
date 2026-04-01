import glob
import json
import os
import re
import shutil
import subprocess

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf


def _ensure_hf_model(model_path: str) -> None:
    """Convert a DeepSpeed ZeRO-3 checkpoint to HF format in-place if needed.

    Checks for a valid config.json with model_type. If absent, runs zero_to_fp32.py
    to consolidate shards. Then always copies HF config/tokenizer files from the
    parent directory (which holds the final HF model saved at the end of training),
    so that the tokenizer chat template is always up to date.
    """
    config_path = os.path.join(model_path, "config.json")
    needs_conversion = True
    if os.path.isfile(config_path):
        with open(config_path) as f:
            if "model_type" in json.load(f):
                needs_conversion = False

    if needs_conversion:
        zero_script = os.path.join(model_path, "zero_to_fp32.py")
        if not os.path.isfile(zero_script):
            raise FileNotFoundError(
                f"No zero_to_fp32.py found in {model_path} and it is not a valid HF model. "
                "Cannot convert to HF format."
            )
        print(f"Converting DeepSpeed checkpoint to HF format: {model_path}")
        subprocess.run(["python3", zero_script, model_path, model_path], check=True)

    # Always copy config and tokenizer files from the parent dir so that the
    # tokenizer chat template (set during training) is present in the checkpoint.
    parent_dir = os.path.dirname(model_path)
    hf_files = [
        "config.json",
        "generation_config.json",
        "chat_template.jinja",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "vocab.json",
    ]
    for fname in hf_files:
        src = os.path.join(parent_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(model_path, fname))


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    # Hydra changes CWD to its output dir; run the subprocess from the repo root.
    repo_root = hydra.utils.get_original_cwd()

    # Derive a checkpoint identifier (e.g. "step_100", "epoch_0") from the model
    # path so that each checkpoint's results go to a distinct subdirectory and
    # can be logged at the correct step in W&B.
    checkpoint_name = os.path.basename(os.path.normpath(cfg.model_path))
    m = re.fullmatch(r"(step|epoch)_(\d+)", checkpoint_name)
    wandb_step = int(m.group(2)) if m else None

    # Use an absolute path so the glob in _log_results_to_wandb works correctly
    # when Hydra changes CWD (e.g. in multirun/SLURM mode).
    output_path = os.path.join(repo_root, cfg.output_root, cfg.eval_suite.name, checkpoint_name)

    _ensure_hf_model(cfg.model_path)

    tasks = ",".join(OmegaConf.to_container(cfg.eval_suite.tasks, resolve=True))

    cmd = [
        "uv", "run", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={cfg.model_path},dtype={cfg.dtype}",
        "--tasks", tasks,
        "--batch_size", str(cfg.batch_size),
        "--output_path", output_path,
    ]

    if cfg.apply_chat_template:
        cmd.append("--apply_chat_template")

    cmd.append("--confirm_run_unsafe_code")

    subprocess.run(cmd, check=True, cwd=repo_root)

    wandb_run_id = OmegaConf.select(cfg, "wandb_run_id")
    wandb_project = OmegaConf.select(cfg, "wandb_project")
    wandb_entity = OmegaConf.select(cfg, "wandb_entity")
    if wandb_run_id and wandb_project:
        _log_results_to_wandb(output_path, wandb_run_id, wandb_project, wandb_entity, wandb_step)


def _log_results_to_wandb(
    output_path: str,
    run_id: str,
    project: str,
    entity: str | None,
    wandb_step: int | None = None,
) -> None:
    json_files = glob.glob(os.path.join(output_path, "**", "results_*.json"), recursive=True)
    if not json_files:
        print(f"No lm_eval results JSON found under {output_path}, skipping W&B logging.")
        return

    results_file = max(json_files, key=os.path.getmtime)
    with open(results_file) as f:
        data = json.load(f)

    task_results = data.get("results", {})

    metrics: dict[str, float] = {}
    for task, task_metrics in task_results.items():
        for metric, value in task_metrics.items():
            if isinstance(value, (int, float)):
                metrics[f"eval/{task}/{metric}"] = value

    if wandb_step is None:
        wandb_step = 0

    metrics["trainer/global_step"] = wandb_step

    with wandb.init(id=run_id, project=project, entity=entity, resume="allow") as run:
        run.log(metrics, step=wandb_step)


if __name__ == "__main__":
    main()
