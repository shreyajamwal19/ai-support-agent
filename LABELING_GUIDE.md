# Labeling Guide

## Current status
`data/golden/golden_set_reviewed.csv` (200 examples) currently holds AI-reviewed labels
(by Claude, not a human) marked `labeling_status = claude_reviewed_not_independent_human`.
These are **not** hand-labels and must not be reported as such.

## How to hand-label them yourself
Run: `python3 scripts/06_review_golden_set.py`
This is a simple menu-driven tool: it shows one message at a time, you type a number to
pick the intent, a number for auto_handle/escalate, and it saves instantly. Full plain-
English instructions are printed when you start it. Your answers go to a separate file
(`data/golden/golden_set_human.csv`) so your old labels are never at risk.

When all 200 are done, run: `python3 scripts/08_finalize_human_labels.py`
This copies your labels into the file the evaluation harness reads, and only then can the
set be called genuinely hand-labelled.

## Sampling methodology (unchanged, already applied)
See `BRAND_SELECTION.md`/`INTENT_TAXONOMY.md`. 200 examples were drawn by five strategies
(see the `sampling_reason` column): proportional stratified (75), rare-intent oversample
(56), short/ambiguous (25), multi-turn context (20), low-confidence boundary (20), plus a
small fill-to-target. Source pool is disjoint from retrieval/training data (leakage-tested).
