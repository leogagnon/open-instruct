"""
Generate a random dataset mixture experiment config.

Reads dataset names from a source experiment YAML (default: balanced.yaml),
assigns each an independent Uniform(0, 1) weight, and writes the result to
a new experiment YAML that can be passed to launch.py.

If a per-source cache directory is available, the mixture is rejected when the
estimated total tokens would be below --min_tokens (default 4e8).

Usage:
    # Write to a temp file and print the path
    python hpo/sft/sample_mixture.py

    # Specify source and output explicitly
    python hpo/sft/sample_mixture.py \
        --source hpo/sft/configs/experiment/balanced.yaml \
        --output hpo/sft/configs/experiment/random_001.yaml \
        --seed 42

    # Launch training immediately after sampling
    python hpo/sft/sample_mixture.py --launch
"""

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import pyarrow as pa
import yaml


MIN_TOKENS_DEFAULT = int(4e8)


def _compute_total_tokens_from_arrow(cache_path: str) -> int:
    """Sum the _token_count column across all Arrow IPC files in *cache_path*."""
    total = 0
    for arrow_file in glob.glob(os.path.join(cache_path, "data-*.arrow")):
        with pa.memory_map(arrow_file, "r") as source:
            reader = pa.ipc.open_stream(source)
            for batch in reader:
                if "_token_count" in batch.schema.names:
                    total += batch.column("_token_count").sum().as_py()
    return total


def _load_token_counts(cache_dir: str) -> dict[str, int]:
    """Scan *cache_dir* for source_meta.json sidecars and return {dataset_name: total_tokens}.

    If a sidecar is missing total_tokens, compute it from the Arrow files and
    persist the result back so subsequent calls are instant.
    """
    counts: dict[str, int] = {}
    if not os.path.isdir(cache_dir):
        return counts
    for entry in os.scandir(cache_dir):
        if not entry.is_dir():
            continue
        meta_path = os.path.join(entry.path, "source_meta.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        name = meta.get("dataset_name")
        if not name:
            continue
        if "total_tokens" not in meta:
            print(f"Computing total_tokens for {name} from Arrow files...")
            total = _compute_total_tokens_from_arrow(entry.path)
            meta["total_tokens"] = total
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        counts[name] = int(meta["total_tokens"])
    return counts


def sample_mixture(source: Path, seed: int | None) -> tuple[list[str], list[float]]:
    rng = random.Random(seed)
    with open(source) as f:
        cfg = yaml.safe_load(f)

    mixer = cfg["dataset_mixer_list"]
    # mixer is alternating [dataset, weight, dataset, weight, ...]
    datasets = mixer[0::2]
    weights = [round(rng.uniform(0, 1), 6) for _ in datasets]
    return datasets, weights


def build_config(name: str, datasets: list[str], weights: list[float]) -> dict:
    mixer = []
    for ds, w in zip(datasets, weights):
        mixer.append(ds)
        mixer.append(w)
    return {"name": name, "dataset_mixer_list": mixer}


def main():
    parser = argparse.ArgumentParser(description="Sample a random dataset mixture config.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("hpo/sft/configs/experiment/full.yaml"),
        help="Source experiment YAML to read dataset names from.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output YAML path. Defaults to hpo/sft/configs/experiment/random_<hex>.yaml.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: random).",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Launch training with the generated config via launch.py after writing it.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="local_dataset_cache",
        help="Per-source cache directory to read token counts from (default: local_dataset_cache).",
    )
    parser.add_argument(
        "--min_tokens",
        type=float,
        default=MIN_TOKENS_DEFAULT,
        help=f"Minimum total tokens required for the mixture (default: {MIN_TOKENS_DEFAULT:.2e}).",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=1,
        help="Number of valid mixture configs to generate (default: 1).",
    )
    args = parser.parse_args()

    token_counts = _load_token_counts(args.cache_dir)
    if not token_counts:
        print(f"No per-source cache found at '{args.cache_dir}'; skipping token check.")

    generated = 0
    attempts = 0
    while generated < args.num:
        seed = args.seed if (args.seed is not None and args.num == 1) else random.randint(0, 0xFFFFFFFF)
        attempts += 1
        datasets, weights = sample_mixture(args.source, seed)

        if token_counts:
            missing = [ds for ds in datasets if ds not in token_counts]
            if missing:
                print(f"Warning: no cached token counts for {len(missing)} source(s); skipping token check.")
                print(f"  Missing: {missing}")
            else:
                estimated_tokens = sum(w * token_counts[ds] for ds, w in zip(datasets, weights))
                if estimated_tokens < args.min_tokens:
                    print(f"  [{attempts}] Rejected: {estimated_tokens:.3e} tokens < {args.min_tokens:.3e}")
                    continue

        hex_suffix = f"{seed & 0xFFFF:04x}"
        name = f"random_{generated}" if args.num > 1 else f"random_{hex_suffix}"

        output = args.output if (args.output and args.num == 1) else Path(f"hpo/sft/configs/experiment/{name}.yaml")
        output.parent.mkdir(parents=True, exist_ok=True)

        cfg = build_config(name, datasets, weights)
        with open(output, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        generated += 1
        estimated_tokens_str = f"{estimated_tokens:.3e}  " if token_counts and not missing else ""
        print(f"[{generated}/{args.num}] Wrote {output}  {estimated_tokens_str}(seed={seed})")
        if args.num == 1:
            print("Weights:")
            for ds, w in zip(datasets, weights):
                print(f"  {w:.4f}  {ds}")

        if args.launch:
            import subprocess

            cmd = [
                "uv", "run", "python", "hpo/sft/launch.py",
                f"experiment={name}",
            ]
            print(f"\nLaunching: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

    if attempts > args.num:
        print(f"\nDone: {generated} configs written ({attempts} attempts, {attempts - generated} rejected).")


if __name__ == "__main__":
    main()
