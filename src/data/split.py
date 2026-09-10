"""Deterministic, leakage-safe split of processed pairs into:
  - retrieval_pool: historical resolutions the system may retrieve/ground on
  - golden_pool: candidate pool the golden eval set is sampled from
  - train_pool: reserved for baseline classifier training (TF-IDF+LogReg intent baseline)

Split is by content_hash (customer message text) with a deterministic hash-based assignment,
NOT random.sample with a seed, so it is 100% reproducible across machines/language versions
and so near-duplicate customer messages (which slipped past exact-dedup, e.g. minor retries)
land in the SAME split instead of leaking across golden/retrieval.
"""
import pandas as pd
import hashlib
from pathlib import Path

IN_PATH = Path("data/processed/pairs.parquet")
OUT_DIR = Path("data/processed")

def bucket(content_hash: str, n_buckets=100) -> int:
    return int(content_hash[:8], 16) % n_buckets

def main():
    df = pd.read_parquet(IN_PATH)
    df["bucket"] = df["content_hash"].apply(bucket)

    # 15% golden candidate pool, 15% held-out train/test for baseline classifier,
    # 70% retrieval pool. Buckets are deterministic hash ranges -> no RNG involved.
    df["split"] = pd.cut(
        df["bucket"], bins=[-1, 14, 29, 99],
        labels=["golden_pool", "baseline_eval_pool", "retrieval_pool"]
    )

    counts = df["split"].value_counts()
    print(counts)

    for name in ["golden_pool", "baseline_eval_pool", "retrieval_pool"]:
        sub = df[df["split"] == name].drop(columns=["bucket"])
        sub.to_parquet(OUT_DIR / f"{name}.parquet", index=False)
        print(f"saved {name}: {len(sub)} rows -> {OUT_DIR / f'{name}.parquet'}")

    # leakage self-check: no content_hash should appear in more than one split
    dup_hashes = df.groupby("content_hash")["split"].nunique()
    n_leak = (dup_hashes > 1).sum()
    print(f"content_hash values spanning >1 split: {n_leak} (should be 0)")
    assert n_leak == 0, "LEAKAGE DETECTED between splits"

if __name__ == "__main__":
    main()
