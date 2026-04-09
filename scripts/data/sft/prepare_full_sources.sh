#!/bin/bash
# Prepares full (pre-subsampling) SFT source datasets and saves them as parquet files.
# Outputs go to ./data/sft_full/ relative to the repo root.
# Run from the repo root: bash scripts/data/sft/prepare_full_sources.sh

set -e

OUTPUT_DIR="./data/sft_full"
mkdir -p "$OUTPUT_DIR"

# Helper: skip a step if the parquet already exists
run_if_missing() {
    local parquet="$1"
    shift
    if [ -f "$parquet" ]; then
        echo "  Skipping — $parquet already exists."
    else
        "$@"
    fi
}

echo "=== WildChat (full GPT-4 subset, ~530k) ==="
run_if_missing "$OUTPUT_DIR/wildchat_gpt4.parquet" \
    python -m scripts.data.sft.wildchat \
        --apply_keyword_filters \
        --apply_empty_message_filters \
        --local_save_dir "$OUTPUT_DIR/wildchat"

echo "=== Aya (full dataset, ~202k) ==="
run_if_missing "$OUTPUT_DIR/aya.parquet" \
    python -m scripts.data.sft.aya \
        --apply_keyword_filters \
        --apply_empty_message_filters \
        --local_save_dir "$OUTPUT_DIR/aya"

echo "=== SciRIFF (full science subset) ==="
run_if_missing "$OUTPUT_DIR/sciriff.parquet" \
    python -m scripts.data.sft.sciriff \
        --apply_empty_message_filters \
        --local_save_dir "$OUTPUT_DIR/sciriff"

echo "=== Table-GPT (full dataset) ==="
run_if_missing "$OUTPUT_DIR/table_gpt.parquet" \
    python -m scripts.data.sft.table_gpt \
        --apply_keyword_filters \
        --apply_empty_message_filters \
        --local_save_dir "$OUTPUT_DIR/table_gpt"

echo "=== FLAN v2 (200k, proportionally scaled from default 90k) ==="
run_if_missing "$OUTPUT_DIR/flan.parquet" \
    python -m scripts.data.sft.flan \
        --apply_keyword_filters \
        --apply_empty_message_filters \
        --total_examples 200000 \
        --local_save_dir "$OUTPUT_DIR/flan"

echo "=== OpenMathInstruct-2 (100k combined GSM8K+MATH) ==="
run_if_missing "$OUTPUT_DIR/open_math_instruct.parquet" \
    python -m scripts.data.sft.open_math_instruct \
        --apply_keyword_filters \
        --apply_empty_message_filters \
        --num_examples 100000 \
        --local_save_dir "$OUTPUT_DIR/open_math_instruct"

echo "=== Evol-CodeAlpaca (full dataset, ~110k) ==="
run_if_missing "$OUTPUT_DIR/evol_codealpaca.parquet" \
    python -m scripts.data.sft.evol_codealpaca \
        --apply_keyword_filters \
        --apply_empty_message_filters \
        --local_save_dir "$OUTPUT_DIR/evol_codealpaca"

echo "=== NuminaMath-TIR (full dataset, ~72k) ==="
run_if_missing "$OUTPUT_DIR/numinamath_tir.parquet" \
    python -m scripts.data.sft.numinamath \
        --apply_keyword_filters \
        --apply_empty_message_filters \
        --local_save_dir "$OUTPUT_DIR/numinamath"

echo "=== Converting Arrow datasets to Parquet ==="
python - <<'EOF'
from datasets import load_from_disk
import os

sources = {
    "wildchat_gpt4": "wildchat_gpt4",
    "aya": "aya",
    "sciriff": "sciriff",
    "table_gpt": "table_gpt",
    "flan": "flan",
    "open_math_instruct": "open_math_instruct",
    "evol_codealpaca": "evol_codealpaca",
    "numinamath_tir": "numinamath_tir",
}

output_dir = "./data/sft_full"
for name, subdir in sources.items():
    parquet_path = os.path.join(output_dir, f"{name}.parquet")
    if os.path.exists(parquet_path):
        print(f"  Skipping {parquet_path} — already exists.")
        continue
    arrow_path = os.path.join(output_dir, subdir)
    print(f"Converting {arrow_path} -> {parquet_path} ...", flush=True)
    ds = load_from_disk(arrow_path)
    if hasattr(ds, "keys"):
        ds = ds["train"]
    print(f"  {len(ds)} examples")
    ds.to_parquet(parquet_path)
    print(f"  Saved to {parquet_path}")

print("Done.")
EOF

echo ""
echo "=== All done. Parquet files in $OUTPUT_DIR: ==="
ls -lh "$OUTPUT_DIR"/*.parquet
