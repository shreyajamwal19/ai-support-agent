# Labeling Guide — Golden Evaluation Set

## Current status: PROVISIONAL, not hand-labeled
`data/golden/golden_set_provisional.{csv,jsonl}` (200 examples) currently carries
`gold_intent` populated from the **rule-based classifier** (`src/intent/rule_classifier.py`)
as a bootstrap, and `gold_action` / `gold_escalation_rationale` are empty. Every row's
`labeling_status` field says `provisional_rule_based_NOT_hand_labeled`. **These are not
human labels and must not be reported as such.** Use `scripts/06_review_golden_set.py`
(a small terminal CLI) to review and correct them; it writes to
`data/golden/golden_set_reviewed.csv` and flips `labeling_status` to `human_reviewed` for
each row you confirm/edit, along with your initials and a timestamp.

## Sampling methodology (already applied, see `BRAND_SELECTION.md`/`INTENT_TAXONOMY.md`)
- Source pool: `data/processed/golden_pool_en.parquet` — a deterministic hash-bucket split
  of AmazonHelp customer↔brand reply pairs, disjoint from `retrieval_pool` and
  `baseline_eval_pool` (0 pair_id or content_hash overlap, checked by `src/data/split.py`
  and `tests/test_split_leakage.py`).
- 200 examples drawn by five explicit strategies (see `sampling_reason` column):
  1. **Proportional stratified** (75): matches the natural intent distribution, capped at
     30 for `OTHER_UNCLEAR` so the set stays informative rather than dominated by the
     catch-all bucket.
  2. **Rare-intent oversample** (56): every intent under 5% natural frequency gets ~8
     forced examples so rare classes are evaluable at all — without this, `ACCOUNT_LOGIN`
     (0.8% of pool) would get ~1-2 examples and any metric on it would be noise.
  3. **Short/ambiguous** (25): messages ≤25 characters — a known hard case with little
     signal for any classifier.
  4. **Multi-turn context** (20): messages with ≥2 prior conversational turns, to stress
     the context-preparation stage.
  5. **Low-confidence boundary** (20): messages the rule classifier itself scored ≤0.4
     confidence — deliberately adversarial to the baseline.

## What a human reviewer should do for each example
1. Read `customer_text_raw` (and `prior_context` if `n_prior_context_turns` > 0).
2. Confirm or correct `gold_intent` against `configs/intent_taxonomy.json` inclusion/
   exclusion criteria. If genuinely ambiguous between two intents, pick the one whose
   inclusion criteria most directly match the customer's primary ask, and note the
   ambiguity in `annotator_notes`.
3. Set `gold_action` to `auto_handle` or `escalate` based on your judgment of what a
   responsible support operation should do — not what the historical brand reply did (the
   brand's real reply is shown for context but may itself have been the wrong call).
4. If `escalate`, set `gold_escalation_rationale` to one of: `insufficient_evidence`,
   `financial_refund`, `account_security`, `abuse_legal_threat`, `novel_unsupported`,
   `low_intent_confidence`, `other` (free text).
5. Set `label_confidence` to `high`/`medium`/`low` — your own confidence in the label, not
   the classifier's.
6. Target labeling throughput: ~200 examples takes a careful annotator roughly 60-90
   minutes at ~20-25s/example: short text, single intent decision, single action decision.

## Inter-point of caution
Do not look at `provisional_intent` before forming your own judgment on hard cases (short/
low-confidence subsets) — anchoring on the baseline's guess defeats the purpose of having
an independent gold label to evaluate that same baseline against.
