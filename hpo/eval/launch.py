import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, List, Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf


@dataclass
class SuiteConfig:
    name: str = MISSING
    tasks: List[str] = field(default_factory=list)


@dataclass
class EvalConfig:
    ckpt_root: str = MISSING
    ckpt: str = MISSING  # format: <wandb_run_id>/step_<n> or <wandb_run_id>/epoch_<n>
    dtype: str = "bfloat16"
    gpus: int = 4
    batch_size: Any = "auto"
    max_model_len: int = 4096
    output_root: str = "output/olmes_eval"

    wandb_project: str = "open_instruct"
    wandb_entity: Optional[str] = None

    # Populated by Hydra group composition (eval_suite=<name>, safety_suite=<name>)
    eval_suite: Any = MISSING
    safety_suite: Optional[Any] = None


cs = ConfigStore.instance()
cs.store(name="eval_schema", node=EvalConfig)


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

    # Remove any .bin files not listed in the weight index (e.g. scheduler.bin,
    # optimizer.bin) so vLLM doesn't try to load them as model weights.
    index_path = os.path.join(model_path, "pytorch_model.bin.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            valid_bins = set(json.load(f)["weight_map"].values())
        for fname in os.listdir(model_path):
            if fname.endswith(".bin") and fname not in valid_bins:
                os.remove(os.path.join(model_path, fname))

    # Convert .bin weights to safetensors. Some vLLM versions prefer safetensors
    # and will download the base model from HuggingFace if not found locally.
    # Always convert when .bin files are present to avoid stale safetensors from
    # a previous HuggingFace download of the base model being used instead.
    has_bin = os.path.isfile(index_path) or any(
        f.endswith(".bin") for f in os.listdir(model_path)
        if not f.startswith("scheduler") and not f.startswith("optimizer")
    )
    if has_bin:
        # Remove stale safetensors index/shards so from_pretrained falls back to .bin.
        for fname in os.listdir(model_path):
            if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
                os.remove(os.path.join(model_path, fname))
        print(f"Converting {model_path} weights to safetensors format...")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.save_pretrained(model_path, safe_serialization=True)
        del model


def _log_results_to_wandb(
    output_path: str,
    run_id: str,
    project: str,
    entity: str | None,
    wandb_step: int | None = None,
) -> None:
    import wandb

    metrics_all_path = os.path.join(output_path, "metrics-all.jsonl")
    if not os.path.isfile(metrics_all_path):
        print(f"No metrics-all.jsonl found under {output_path}, skipping W&B logging.")
        return

    metrics: dict[str, float] = {}
    with open(metrics_all_path) as f:
        for line in f:
            task_data = json.loads(line)
            task_name = task_data["task_name"]
            for metric, value in task_data.get("metrics", {}).items():
                if isinstance(value, (int, float)):
                    metrics[f"eval/{task_name}/{metric}"] = value

    if wandb_step is None:
        wandb_step = 0

    metrics["trainer/global_step"] = wandb_step

    with wandb.init(id=run_id, project=project, entity=entity, resume="allow") as run:
        run.define_metric("trainer/global_step")
        run.define_metric("eval/*", step_metric="trainer/global_step")
        run.log(metrics)


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg: EvalConfig) -> None:
    # Hydra changes CWD to its output dir; run the subprocess from the repo root.
    repo_root = hydra.utils.get_original_cwd()
    olmes_root = os.path.join(repo_root, "olmes")
    python_bin = os.path.join(olmes_root, ".venv", "bin", "python")

    # ckpt is "<wandb_run_id>/step_<n>" or "<wandb_run_id>/epoch_<n>".
    # Split into run ID and checkpoint name for W&B logging and output paths.
    ckpt_parts = cfg.ckpt.split("/", 1)
    if len(ckpt_parts) != 2:
        raise ValueError(f"ckpt must be in the format '<wandb_run_id>/step_<n>', got: {cfg.ckpt!r}")
    wandb_run_id, checkpoint_name = ckpt_parts
    model_path = os.path.join(cfg.ckpt_root, cfg.ckpt)

    m = re.fullmatch(r"(step|epoch)_(\d+)", checkpoint_name)
    wandb_step = int(m.group(2)) if m else None

    output_path = os.path.join(repo_root, cfg.output_root, cfg.eval_suite.name, checkpoint_name)

    _ensure_hf_model(model_path)

    # safety-eval imports openai at module level, requiring OPENAI_API_KEY even for
    # local-only classifiers. Set a dummy key so the import doesn't crash.
    # Clear PYTHONPATH so CVMFS system packages don't shadow the venv's packages.
    olmes_env = os.environ.copy()
    olmes_env.setdefault("OPENAI_API_KEY", "dummy")
    olmes_env["PYTHONPATH"] = ""

    tasks = OmegaConf.to_container(cfg.eval_suite.tasks, resolve=True)

    cmd = [
        python_bin, "-m", "oe_eval.launch",
        "--model", model_path,
        "--model-type", "vllm",
        "--model-args", json.dumps({"dtype": cfg.dtype, "trust_remote_code": False, "max_length": cfg.max_model_len}),
        "--task", *tasks,
        "--output-dir", output_path,
        "--gpus", str(cfg.gpus),
        "--batch-size", str(cfg.batch_size),
    ]

    subprocess.run(cmd, check=True, cwd=olmes_root, env=olmes_env)

    wandb_project = OmegaConf.select(cfg, "wandb_project")
    wandb_entity = OmegaConf.select(cfg, "wandb_entity")
    if wandb_project:
        _log_results_to_wandb(output_path, wandb_run_id, wandb_project, wandb_entity, wandb_step)

    # Run safety suite separately if configured (external evals cannot be mixed with regular tasks).
    safety_suite = OmegaConf.select(cfg, "safety_suite")
    if safety_suite:
        safety_tasks = OmegaConf.to_container(safety_suite.tasks, resolve=True)
        safety_output_path = os.path.join(
            repo_root, cfg.output_root, safety_suite.name, checkpoint_name
        )
        safety_cmd = [
            python_bin, "-m", "oe_eval.launch",
            "--model", model_path,
            "--model-type", "vllm",
            "--model-args", json.dumps({"dtype": cfg.dtype, "trust_remote_code": False, "max_length": cfg.max_model_len}),
            "--task", *safety_tasks,
            "--output-dir", safety_output_path,
            "--gpus", str(cfg.gpus),
            "--batch-size", str(cfg.batch_size),
        ]
        subprocess.run(safety_cmd, check=True, cwd=olmes_root, env=olmes_env)
        if wandb_project:
            _log_results_to_wandb(
                safety_output_path, wandb_run_id, wandb_project, wandb_entity, wandb_step
            )


if __name__ == "__main__":
    main()
