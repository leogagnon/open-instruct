import dataclasses
import glob
import os
import random
import subprocess
from datetime import datetime

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.finetune import FlatArguments


def _to_cli_args(obj) -> list[str]:
    """Serialize a dataclass to CLI flag pairs, skipping None and default-valued fields."""
    cls = type(obj)
    defaults = {}
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            defaults[f.name] = f.default_factory()

    args = []
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        value = getattr(obj, f.name)
        if value is None:
            continue
        if f.name in defaults and value == defaults[f.name]:
            continue
        flag = f"--{f.name}"
        if isinstance(value, bool):
            args += [flag, str(value)]
        elif isinstance(value, list):
            args += [flag] + [str(v) for v in value]
        else:
            args += [flag, str(value)]
    return args


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    # Hydra changes CWD to its output dir; run the subprocess from the repo root.
    repo_root = hydra.utils.get_original_cwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand_suffix = f"{random.randint(0, 0xFFFF):04x}"
    unique_name = f"{cfg.experiment.name}_{timestamp}_{rand_suffix}"

    # Pre-generate a W&B run ID so both training and eval can share the same run.
    # The run ID also serves as the output directory name (mirrors nanochat's logs/results/<id>/ layout).
    wandb_run_id = wandb.util.generate_id()
    output_dir = os.path.join(repo_root, cfg.output_root, wandb_run_id)
    wandb_project = OmegaConf.select(cfg, "wandb_project") or "open_instruct_internal"
    # Resolve entity: explicit config > WANDB_ENTITY env var > wandb default (None).
    wandb_entity = OmegaConf.select(cfg, "wandb_entity") or os.environ.get("WANDB_ENTITY")

    # Build typed argument objects — field names are validated against the dataclass
    # definitions at construction time, so renames in finetune.py fail loudly here.
    train_args = FlatArguments(
        exp_name=unique_name,
        model_name_or_path=cfg.model_name_or_path,
        model_revision="main",
        max_seq_length=cfg.max_seq_length,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        checkpointing_steps=str(cfg.checkpointing_steps),
        keep_last_n_checkpoints=cfg.keep_last_n_checkpoints,
        clean_checkpoints_at_end=cfg.clean_checkpoints_at_end,
        save_initial_checkpoint=cfg.save_initial_checkpoint,
        output_dir=output_dir,
        do_not_randomize_output_dir=True,
        report_to="wandb",
        with_tracking=True,
        wandb_project_name=wandb_project,
        logging_steps=1,
        seed=cfg.seed,
        push_to_hub=False,
        try_launch_beaker_eval_jobs=False,
        dataset_mixer_list=[str(x) for x in cfg.experiment.dataset_mixer_list],
        max_train_steps=OmegaConf.select(cfg, "max_train_steps"),
        dataset_skip_cache=cfg.dataset_skip_cache,
        cache_dataset_only=cfg.cache_dataset_only,
        dataset_local_cache_dir=cfg.dataset_local_cache_dir,
        dataset_use_per_source_cache=cfg.dataset_use_per_source_cache,
    )
    tc = TokenizerConfig(
        tokenizer_name_or_path=cfg.tokenizer_name,
        tokenizer_name=cfg.tokenizer_name,  # keeps hash compatible with existing caches
        tokenizer_revision="main",
        add_bos=True,
        chat_template_name="tulu",
    )

    # Save the resolved config next to the checkpoints for reproducibility.
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    cmd = [
        "uv", "run", "accelerate", "launch",
        "--mixed_precision", "bf16",
        "--num_processes", str(cfg.num_processes),
        "--num_machines", "1",
        "--dynamo_backend", "no",
        "--use_deepspeed",
        "--deepspeed_config_file", "configs/ds_configs/stage3_no_offloading_accelerate.conf",
        "open_instruct/finetune.py",
    ] + _to_cli_args(train_args) + _to_cli_args(tc)

    # Inject the pre-generated run ID so finetune.py's wandb run uses it.
    # Also pin WANDB_ENTITY so lm_eval can resume the exact same run.
    env = os.environ.copy()
    env["WANDB_RUN_ID"] = wandb_run_id
    if wandb_entity:
        env["WANDB_ENTITY"] = wandb_entity
    sweep_id = OmegaConf.select(cfg, "sweep_id")
    if sweep_id:
        env["WANDB_RUN_GROUP"] = str(sweep_id)

    subprocess.run(cmd, check=True, cwd=repo_root, env=env)

    if OmegaConf.select(cfg, "eval.enabled") and cfg.eval.enabled:
        if OmegaConf.select(cfg, "eval.eval_suite") is None:
            raise ValueError("eval.eval_suite must be set when eval.enabled=true")

        # Find all saved checkpoints (step_N / epoch_N); fall back to the final
        # output dir if none exist (e.g. checkpointing was disabled).
        checkpoint_dirs = sorted(
            glob.glob(os.path.join(output_dir, "step_*"))
            + glob.glob(os.path.join(output_dir, "epoch_*"))
        )
        if not checkpoint_dirs:
            checkpoint_dirs = [output_dir]

        # Hydra multirun accepts comma-separated values for a single override;
        # each becomes a separate SLURM job and they run in parallel up to
        # array_parallelism (32 by default in the fir launcher config).
        model_paths = ",".join(checkpoint_dirs)

        eval_script = os.path.join(repo_root, "hpo/eval/launch.py")
        eval_cmd = [
            "uv", "run", "python", eval_script,
            "-m",  # multirun: submits to SLURM via the configured launcher
            f"model_path={model_paths}",
            f"eval_suite={cfg.eval.eval_suite}",
            f"wandb_run_id={wandb_run_id}",
            f"wandb_project={wandb_project}",
        ]
        if wandb_entity:
            eval_cmd.append(f"wandb_entity={wandb_entity}")
        subprocess.run(eval_cmd, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
