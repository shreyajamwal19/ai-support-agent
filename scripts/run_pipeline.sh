#!/usr/bin/env bash
# Full reproduction: raw twcs.csv -> processed data -> golden set -> models/index -> eval.
# Measured end-to-end on this dev machine: ~8-9 minutes (excludes pip install).
set -e
cd "$(dirname "$0")/.."

echo "[1/8] Dataset overview + brand selection evidence..."
python3 scripts/01_dataset_overview.py
python3 scripts/02_brand_selection.py
python3 scripts/03_deflection_check.py

echo "[2/8] Conversation reconstruction + dedup..."
python3 src/data/reconstruct.py

echo "[3/8] Deterministic leakage-safe splits..."
python3 src/data/split.py

echo "[4/8] English-language filtering..."
python3 -c "
import sys; sys.path.insert(0,'.')
import pandas as pd
from src.data.lang_filter import is_english
for name in ['golden_pool','baseline_eval_pool','retrieval_pool']:
    df = pd.read_parquet(f'data/processed/{name}.parquet')
    mask = df['customer_text_clean'].apply(is_english)
    df[mask].reset_index(drop=True).to_parquet(f'data/processed/{name}_en.parquet', index=False)
    print(name, len(df), '->', mask.sum())
"

echo "[5/8] Taxonomy clustering evidence (informational, taxonomy already finalized)..."
python3 scripts/04_taxonomy_clustering.py > /tmp/taxonomy_clustering.log 2>&1 || true

echo "[6/8] Golden set build (regenerates PROVISIONAL labels only -- reviewed labels in git are preserved separately)..."
python3 scripts/05_build_golden_set.py

echo "[7/8] Fit retriever + train baselines..."
python3 src/retrieval/tfidf_retriever.py > /tmp/retriever.log 2>&1
python3 src/intent/baselines.py

echo "[8/8] Run evaluation harness..."
python3 -m src.evaluation.run --config configs/eval.yaml

echo ""
echo "Done. See artifacts/eval_results/summary.txt and results.json"
