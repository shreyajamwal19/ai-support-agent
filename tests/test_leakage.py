"""Phase 13/18: leakage safeguards. These tests FAIL the build if evaluation data leaks
into retrieval/training data."""
import pandas as pd
import pytest
from pathlib import Path

DATA = Path("data/processed")


@pytest.fixture(scope="module")
def splits():
    if not (DATA / "golden_pool_en.parquet").exists():
        pytest.skip("processed data not present (run scripts/01-05 first, requires data/raw/twcs.csv)")
    return {
        "golden": pd.read_parquet(DATA / "golden_pool_en.parquet"),
        "baseline_eval": pd.read_parquet(DATA / "baseline_eval_pool_en.parquet"),
        "retrieval": pd.read_parquet(DATA / "retrieval_pool_en.parquet"),
    }


def test_no_pair_id_overlap_golden_retrieval(splits):
    overlap = set(splits["golden"]["pair_id"]) & set(splits["retrieval"]["pair_id"])
    assert len(overlap) == 0, f"Leakage: {len(overlap)} pair_ids in both golden and retrieval pools"


def test_no_pair_id_overlap_golden_baseline(splits):
    overlap = set(splits["golden"]["pair_id"]) & set(splits["baseline_eval"]["pair_id"])
    assert len(overlap) == 0


def test_no_content_hash_overlap_golden_retrieval(splits):
    overlap = set(splits["golden"]["content_hash"]) & set(splits["retrieval"]["content_hash"])
    assert len(overlap) == 0, "Near-duplicate leakage via content_hash between golden and retrieval pools"


def test_golden_set_pairs_excluded_from_actual_retrieval_index():
    """Verifies the golden CSV pair_ids used for evaluation are not queryable hits in the
    fitted retriever's own pool (belt-and-suspenders on top of the split-level check)."""
    golden_path = Path("data/golden/golden_set_provisional.csv")
    retriever_pkl = Path("artifacts/index/tfidf_retriever.pkl")
    if not golden_path.exists() or not retriever_pkl.exists():
        pytest.skip("golden set or fitted retriever not present")
    import pickle
    golden = pd.read_csv(golden_path)
    with open(retriever_pkl, "rb") as f:
        idx = pickle.load(f)
    overlap = set(golden["pair_id"]) & set(idx["df"]["pair_id"])
    assert len(overlap) == 0
