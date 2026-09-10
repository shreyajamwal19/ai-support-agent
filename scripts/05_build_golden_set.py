"""Phase 4: build the golden evaluation set candidate pool with a defensible, stratified,
reproducible sampling strategy. This produces PROVISIONAL labels (rule-classifier +
heuristics) for bootstrapping -- these are explicitly NOT hand-labels. A human (the
assignment author) must review/correct them using LABELING_GUIDE.md before the set can be
called "hand-labelled" in the report. See labeling_status field on every row.
"""
import pandas as pd
import numpy as np
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, ".")
from src.intent.rule_classifier import classify, INTENT_IDS

SEED = 42
TARGET_N = 200

df = pd.read_parquet("data/processed/golden_pool_en.parquet")
print(f"Golden candidate pool: {len(df)} rows")

# Run rule classifier to get provisional intent + stratification signal
prov = df["customer_text_clean"].apply(classify)
df["provisional_intent"] = prov.apply(lambda d: d["intent"])
df["provisional_confidence"] = prov.apply(lambda d: d["confidence"])
df["char_len"] = df["customer_text_clean"].str.len()

print(df["provisional_intent"].value_counts())

rng = np.random.RandomState(SEED)

# Stratified sampling design:
#  - ~55% proportional-to-frequency across intents (captures realistic distribution)
#  - ~20% deliberately oversampled from RARE intents (so rare classes are evaluable at all)
#  - ~15% short messages (<=25 chars) -- known hard case (little signal)
#  - ~10% messages with prior_context turns > 0 -- multi-turn cases
# Target ~200 total, dedup across categories by pair_id.

chosen_ids = set()
rows = []

def take(pool, n, tag):
    pool = pool[~pool["pair_id"].isin(chosen_ids)]
    n = min(n, len(pool))
    picked = pool.sample(n=n, random_state=rng.randint(0, 1_000_000)) if n > 0 else pool.iloc[0:0]
    for _, r in picked.iterrows():
        chosen_ids.add(r["pair_id"])
        rows.append((r, tag))
    return len(picked)

# 1. Proportional stratified sample across all intents (main body, ~110 examples).
# OTHER_UNCLEAR dominates the raw pool (66%) because the rule classifier is a coarse
# keyword matcher with a broad catch-all -- see COMPLAINT bucket below. We deliberately
# cap its share of the golden set (rather than sample it proportionally) so the set stays
# informative for evaluating the *named* intents, which is the point of a golden set.
# This is a stratification design choice, documented in DECISIONS.md.
intent_counts = df["provisional_intent"].value_counts()
CAP_OTHER = 30
props = (intent_counts / intent_counts.sum() * 110).round().astype(int).clip(lower=3)
props["OTHER_UNCLEAR"] = min(props.get("OTHER_UNCLEAR", CAP_OTHER), CAP_OTHER)
for intent_id, n in props.items():
    sub = df[df["provisional_intent"] == intent_id]
    take(sub, n, "stratified_proportional")

# 2. Rare-intent oversample: any intent with <5% frequency, add up to 8 more each
rare_intents = intent_counts[intent_counts / intent_counts.sum() < 0.05].index
for intent_id in rare_intents:
    sub = df[df["provisional_intent"] == intent_id]
    take(sub, 8, "rare_intent_oversample")

# 3. Short/ambiguous messages (hard cases for any classifier)
short_pool = df[df["char_len"] <= 25]
take(short_pool, 25, "short_ambiguous")

# 4. Multi-turn context cases
multiturn_pool = df[df["n_prior_context_turns"] >= 2]
take(multiturn_pool, 20, "multi_turn_context")

# 5. Low provisional-confidence cases (likely ambiguous / boundary)
lowconf_pool = df[df["provisional_confidence"] <= 0.4]
take(lowconf_pool, 20, "low_confidence_boundary")

# Trim/pad to target
golden = pd.DataFrame([r for r, tag in rows])
tags = [tag for r, tag in rows]
golden["sampling_reason"] = tags
if len(golden) > 250:
    golden = golden.sample(n=250, random_state=SEED).reset_index(drop=True)
elif len(golden) < TARGET_N:
    remaining = df[~df["pair_id"].isin(golden["pair_id"])]
    extra_n = TARGET_N - len(golden)
    extra = remaining.sample(n=min(extra_n, len(remaining)), random_state=SEED)
    extra = extra.copy()
    extra["sampling_reason"] = "fill_to_target_random"
    golden = pd.concat([golden, extra], ignore_index=True)

golden = golden.drop_duplicates(subset=["pair_id"]).reset_index(drop=True)
golden["example_id"] = [f"GOLD_{i:04d}" for i in range(len(golden))]
golden["labeling_status"] = "provisional_rule_based_NOT_hand_labeled"
golden["gold_intent"] = golden["provisional_intent"]  # placeholder until human review
golden["gold_action"] = None
golden["gold_escalation_rationale"] = None
golden["label_confidence"] = None
golden["annotator_notes"] = ""

print(f"\nFinal golden set size: {len(golden)}")
print(golden["sampling_reason"].value_counts())
print(golden["provisional_intent"].value_counts())

out_cols = ["example_id","pair_id","customer_tweet_id","customer_text_raw","customer_text_clean",
            "prior_context","n_prior_context_turns","brand_reply_raw","provisional_intent",
            "provisional_confidence","sampling_reason","labeling_status","gold_intent",
            "gold_action","gold_escalation_rationale","label_confidence","annotator_notes"]
golden[out_cols].to_json("data/golden/golden_set_provisional.jsonl", orient="records", lines=True)
golden[out_cols].to_csv("data/golden/golden_set_provisional.csv", index=False)
print("Saved data/golden/golden_set_provisional.{jsonl,csv}")

# leakage check vs retrieval pool
retrieval = pd.read_parquet("data/processed/retrieval_pool_en.parquet")
overlap = set(golden["pair_id"]) & set(retrieval["pair_id"])
print(f"\nLEAKAGE CHECK -- golden pair_ids present in retrieval pool: {len(overlap)} (must be 0)")
assert len(overlap) == 0
hash_overlap = set(golden["pair_id"].apply(lambda x: x)) # noop, hash check below
