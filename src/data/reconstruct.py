"""Phase 2: reconstruct AmazonHelp customer<->brand conversation pairs from raw TWCS csv.

A "pair" = one customer (inbound) tweet + the single AmazonHelp reply that is a *direct*
response to it (in_response_to_tweet_id -> tweet_id). We do not attempt to reconstruct full
N-turn threads beyond capturing up to 3 hops of prior context, because TWCS threading is a
simple reply-chain (not a DAG with branching resolution paths) and deeper reconstruction adds
complexity without adding grounding value for this assignment's scope.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import re
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BRAND = "AmazonHelp"
RAW_PATH = Path("data/raw/twcs.csv")
OUT_PATH = Path("data/processed/pairs.parquet")

MENTION_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"http[s]?://\S+")
WS_RE = re.compile(r"\s+")


def clean_text_preserve(text: str) -> str:
    """Light normalization. Preserves original text separately for auditability -- this
    function only produces the *normalized* copy used for modeling/matching."""
    if not isinstance(text, str):
        return ""
    t = URL_RE.sub("[URL]", text)
    t = MENTION_RE.sub("", t)  # strip @handles (brand + customer @mentions), keep content
    t = WS_RE.sub(" ", t).strip()
    return t


def content_hash(text: str) -> str:
    return hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()[:16]


def get_context(by_id: pd.DataFrame, start_id, max_hops=3):
    """Walk backwards from a customer tweet through in_response_to_tweet_id to collect up to
    max_hops prior turns as conversational context. Returns oldest-first list of texts."""
    ctx = []
    cur = start_id
    hops = 0
    seen = set()
    while hops < max_hops and pd.notna(cur) and cur not in seen:
        seen.add(cur)
        if cur not in by_id.index:
            break
        row = by_id.loc[cur]
        ctx.append({"tweet_id": int(cur), "author_id": row["author_id"],
                     "inbound": bool(row["inbound"]), "text": row["text"]})
        cur = row["in_response_to_tweet_id"]
        hops += 1
    return list(reversed(ctx))


def build_pairs() -> pd.DataFrame:
    log.info("Loading raw csv...")
    df = pd.read_csv(RAW_PATH, dtype={"tweet_id": "int64", "author_id": str}, low_memory=False)
    df["in_response_to_tweet_id"] = pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce")
    by_id = df.set_index("tweet_id", drop=False)

    brand_rows = df[(df["inbound"] == False) & (df["author_id"] == BRAND)].copy()
    parent_ids = brand_rows["in_response_to_tweet_id"]
    valid_mask = parent_ids.notna() & parent_ids.astype("Int64").isin(by_id.index)
    brand_rows = brand_rows[valid_mask]
    parent_ids = parent_ids[valid_mask].astype("int64")

    parents = by_id.loc[parent_ids.values]
    cust_mask = (parents["inbound"] == True).values
    brand_rows = brand_rows[cust_mask]
    parent_ids = parent_ids[cust_mask]
    parents = parents[cust_mask]

    log.info(f"Found {len(brand_rows)} candidate customer->{BRAND} reply pairs")

    records = []
    for (b_idx, b_row), (c_tweet_id) in zip(brand_rows.iterrows(), parent_ids.values):
        c_row = by_id.loc[c_tweet_id]
        context = get_context(by_id, c_row["in_response_to_tweet_id"], max_hops=3)
        records.append({
            "pair_id": f"{BRAND}_{c_tweet_id}_{b_row['tweet_id']}",
            "customer_tweet_id": int(c_tweet_id),
            "brand_reply_tweet_id": int(b_row["tweet_id"]),
            "customer_text_raw": c_row["text"],
            "customer_text_clean": clean_text_preserve(c_row["text"]),
            "brand_reply_raw": b_row["text"],
            "brand_reply_clean": clean_text_preserve(b_row["text"]),
            "customer_created_at": c_row["created_at"],
            "brand_created_at": b_row["created_at"],
            "prior_context": context,  # list of dicts, oldest-first
            "n_prior_context_turns": len(context),
        })

    out = pd.DataFrame.from_records(records)
    out["content_hash"] = out["customer_text_clean"].apply(content_hash)

    n_before = len(out)
    # exact-duplicate customer messages (templated/spam/retries): keep first occurrence only
    out = out.drop_duplicates(subset=["content_hash"], keep="first").reset_index(drop=True)
    n_after = len(out)
    log.info(f"Deduplicated {n_before - n_after} exact-duplicate customer messages "
              f"({100*(n_before-n_after)/n_before:.1f}%)")

    # drop pairs with empty/near-empty customer text after cleaning (junk rows)
    out = out[out["customer_text_clean"].str.len() >= 3].reset_index(drop=True)
    log.info(f"Final pair count: {len(out)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    log.info(f"Saved {OUT_PATH}")
    return out


if __name__ == "__main__":
    build_pairs()
