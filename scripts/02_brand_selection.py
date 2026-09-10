"""Phase 1: compare candidate brands on measurable criteria, pick one with evidence."""
import pandas as pd
import numpy as np
import json, re
from pathlib import Path

RAW = Path("data/raw/twcs.csv")
OUT = Path("artifacts/data_quality")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW, dtype={"tweet_id": "int64", "author_id": str}, low_memory=False)
df["in_response_to_tweet_id"] = pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce")

# index tweets by id for chain walking
by_id = df.set_index("tweet_id", drop=False)

candidates = ["AmazonHelp","AppleSupport","Uber_Support","SpotifyCares","Delta",
              "AmericanAir","TMobileHelp","comcastcares","British_Airways","SouthwestAir",
              "AskPlayStation","XboxSupport","hulu_support","VerizonSupport"]

def analyze_brand(handle):
    brand_rows = df[(df["inbound"] == False) & (df["author_id"] == handle)]
    # brand replies that are direct responses to a customer (inbound) tweet
    parent_ids = brand_rows["in_response_to_tweet_id"].dropna().astype("int64")
    parents = by_id.reindex(parent_ids)
    valid_pairs = parents[parents["inbound"] == True]
    n_pairs = len(valid_pairs)

    # of those customer tweets, how many are thread-roots (no further parent) = fresh complaints
    is_root = valid_pairs["in_response_to_tweet_id"].isna().sum()

    # thread length: walk back from brand reply to root, capped at 10 hops
    def thread_len(tweet_id, hops=0):
        if hops >= 10 or pd.isna(tweet_id):
            return hops
        row = by_id.loc[tweet_id] if tweet_id in by_id.index else None
        if row is None or pd.isna(row.get("in_response_to_tweet_id")):
            return hops
        return thread_len(row["in_response_to_tweet_id"], hops + 1)

    sample_ids = brand_rows["tweet_id"].sample(min(300, len(brand_rows)), random_state=42) if len(brand_rows) else pd.Series([], dtype="int64")
    lens = [thread_len(t) for t in sample_ids]
    avg_len = float(np.mean(lens)) if lens else 0.0

    cust_texts = valid_pairs["text"].dropna()
    avg_char_len = float(cust_texts.str.len().mean()) if len(cust_texts) else 0.0
    url_rate = float(cust_texts.str.contains(r"http[s]?://", regex=True).mean()) if len(cust_texts) else 0.0
    dup_rate = float(1 - (cust_texts.nunique() / max(len(cust_texts),1)))

    return {
        "brand": handle,
        "n_brand_msgs_total": int(len(brand_rows)),
        "n_customer_brand_reply_pairs": int(n_pairs),
        "n_fresh_root_complaints": int(is_root),
        "pct_fresh_root_complaints": round(100*is_root/max(n_pairs,1), 1),
        "avg_thread_depth_sampled": round(avg_len, 2),
        "avg_customer_msg_char_len": round(avg_char_len, 1),
        "customer_msg_url_rate_pct": round(100*url_rate, 1),
        "customer_msg_dup_rate_pct": round(100*dup_rate, 1),
    }

results = [analyze_brand(h) for h in candidates]
res_df = pd.DataFrame(results).sort_values("n_customer_brand_reply_pairs", ascending=False)
print(res_df.to_string(index=False))
res_df.to_csv(OUT / "brand_candidates.csv", index=False)
res_df.to_json(OUT / "brand_candidates.json", orient="records", indent=2)
print("\nSaved brand_candidates.csv/json")
