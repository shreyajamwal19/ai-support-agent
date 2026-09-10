"""Phase 7: lightweight TF-IDF nearest-neighbor retrieval over historical resolved
AmazonHelp conversations. Deliberately NOT a heavy vector DB -- appropriate scope for a
take-home: a fit TF-IDF matrix + cosine similarity via sklearn is fast, fully local,
inspectable, and sufficient at this corpus size (~79k English retrieval-pool pairs).

Critically: the fit corpus is `retrieval_pool_en.parquet`, which is disjoint (by
content_hash and pair_id) from `golden_pool` and `baseline_eval_pool` by construction in
src/data/split.py. tests/test_retrieval_no_leakage.py asserts this at import/build time.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import pickle
import logging

log = logging.getLogger(__name__)

INDEX_DIR = Path("artifacts/index")


class TfidfRetriever:
    def __init__(self, pool_path="data/processed/retrieval_pool_en.parquet"):
        self.pool_path = pool_path
        self.df = None
        self.vectorizer = None
        self.matrix = None

    def fit(self):
        self.df = pd.read_parquet(self.pool_path).reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(max_features=20000, stop_words="english",
                                           ngram_range=(1, 2), min_df=2)
        self.matrix = self.vectorizer.fit_transform(self.df["customer_text_clean"])
        log.info(f"Fit retriever on {len(self.df)} historical pairs, "
                 f"vocab size {len(self.vectorizer.vocabulary_)}")
        return self

    def save(self, path=INDEX_DIR):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "tfidf_retriever.pkl", "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "matrix": self.matrix,
                         "df": self.df}, f)

    def load(self, path=INDEX_DIR):
        path = Path(path)
        with open(path / "tfidf_retriever.pkl", "rb") as f:
            d = pickle.load(f)
        self.vectorizer, self.matrix, self.df = d["vectorizer"], d["matrix"], d["df"]
        return self

    def query(self, text: str, k: int = 3, min_similarity: float = 0.15,
               exclude_pair_ids: set | None = None) -> list[dict]:
        """Returns up to k historical (customer_msg, brand_resolution) evidence items with
        cosine similarity >= min_similarity, sorted descending. Empty list = no sufficiently
        similar historical case (system should treat this as low grounding evidence)."""
        if self.matrix is None:
            raise RuntimeError("Retriever not fit/loaded")
        qvec = self.vectorizer.transform([text])
        sims = cosine_similarity(qvec, self.matrix)[0]
        order = np.argsort(-sims)
        results = []
        for idx in order[:50]:
            if len(results) >= k:
                break
            score = float(sims[idx])
            if score < min_similarity:
                break
            row = self.df.iloc[idx]
            if exclude_pair_ids and row["pair_id"] in exclude_pair_ids:
                continue
            results.append({
                "pair_id": row["pair_id"],
                "relevance": round(score, 3),
                "historical_customer_message": row["customer_text_clean"],
                "historical_resolution": row["brand_reply_clean"],
            })
        return results


if __name__ == "__main__":
    r = TfidfRetriever().fit()
    r.save()
    for q in ["where is my order it hasn't arrived",
              "I want a refund for this damaged item",
              "can't log in my password isn't working"]:
        print(f"\nQuery: {q}")
        for hit in r.query(q, k=3):
            print(f"  sim={hit['relevance']} | {hit['historical_customer_message'][:80]!r}")
            print(f"    -> {hit['historical_resolution'][:100]!r}")
