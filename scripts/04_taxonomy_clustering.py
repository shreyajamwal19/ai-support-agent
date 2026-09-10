"""Phase 3 evidence step (v2, English-filtered): TF-IDF + KMeans over English customer
messages to discover candidate intent clusters from real data."""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
import json
from pathlib import Path

df = pd.read_parquet("data/processed/baseline_eval_pool_en.parquet")
texts = df["customer_text_clean"].tolist()
print(f"Clustering over {len(texts)} English customer messages")

vec = TfidfVectorizer(max_features=6000, stop_words="english", ngram_range=(1,2), min_df=8)
X = vec.fit_transform(texts)

k = 12
km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10, batch_size=1000)
labels = km.fit_predict(X)

terms = np.array(vec.get_feature_names_out())
report = {}
for c in range(k):
    center = km.cluster_centers_[c]
    top_idx = center.argsort()[::-1][:12]
    top_terms = terms[top_idx].tolist()
    size = int((labels == c).sum())
    examples = df.loc[labels == c, "customer_text_clean"].head(5).tolist()
    report[c] = {"size": size, "top_terms": top_terms, "example_messages": examples}
    print(f"\nCluster {c} (n={size}): {top_terms}")
    for e in examples:
        print("   -", e[:130])

Path("artifacts/data_quality").mkdir(parents=True, exist_ok=True)
with open("artifacts/data_quality/taxonomy_clusters_en.json", "w") as f:
    json.dump(report, f, indent=2)
