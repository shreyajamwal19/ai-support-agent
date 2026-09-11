# AI Support Agent — AmazonHelp (Customer Support on Twitter)

A grounded intent-classification + retrieval + escalation system for AmazonHelp, built on
the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset (TWCS), plus a full evaluation framework designed so that **every number in
`REPORT.md` is honestly earned and reproducible** — see especially `REPORT.md` §9,
"What Is Misleading About My Headline Number," before trusting any metric.

> **This assignment's own framing — "the proof is worth more than the system" — is why
> this README leads with reproducibility and honesty sections instead of a feature list.**

## 1. One-paragraph summary

Given a customer tweet directed at AmazonHelp, the system classifies its intent (12-class
taxonomy derived from real clustering of the data, not assumed), retrieves the most
similar historical customer issue + AmazonHelp's actual resolution via TF-IDF, drafts a
reply grounded in that historical resolution, runs a regex-based safety check to catch
fabricated promises, and applies an explicit multi-signal risk policy to decide
auto-handle vs. escalate-to-human with a stated reason. An LLM-based classification/
generation/judge path is implemented against the real Anthropic API but **was not
executed for this submission's reported numbers** — no API key was available in the build
sandbox. This is disclosed prominently, not buried (`DECISIONS.md` #9, `REPORT.md` §9).

## 2. Architecture

```
customer message
   -> conversation/context prep (src/data/reconstruct.py)
   -> intent classification (src/intent/{rule_classifier,baselines}.py)
   -> historical resolution retrieval (src/retrieval/tfidf_retriever.py)
   -> response drafting (src/generation/responder.py; LLM path: llm_responder.py, unexecuted)
   -> grounding/safety check (regex promise-detector)
   -> escalation decision (src/escalation/policy.py, 7 named risk signals, OR-logic)
   -> structured JSON output (src/pipeline/agent.py)
   -> evaluation/logging (src/evaluation/run.py)
```

## 3. Key design decisions (full list in `DECISIONS.md`, 15 entries)

- **Brand: AmazonHelp**, chosen on measured evidence (0.6% DM-deflection rate vs.
  16.5-71.5% for every other high-volume candidate — see `BRAND_SELECTION.md`), not
  convenience.
- **TF-IDF retrieval, not a vector DB** — appropriate for ~79k-document scale, fully
  local, inspectable.
- **Escalation is 7 named risk signals (OR-logic), not `if confidence < X`** — biased
  toward escalating because false auto-handling is worse than an unnecessary escalation.
- **No LLM calls in the reported numbers** — no API key in the build sandbox. Every
  classification/generation/grounding number below comes from regex rules + TF-IDF. This
  is the single most important scope caveat in this repository.

## 4. Setup

```bash
git clone https://github.com/shreyajamwal19/ai-support-agent.git
cd ai-support-agent
pip install -r requirements.txt --break-system-packages   # or use a venv
cp .env.example .env   # optional: only needed for the LLM path
```

Python 3.10+ recommended (built/tested on 3.12).

## 5. Dataset

Kaggle's dataset host is not reachable from every sandboxed environment (it wasn't from
ours — see `PROJECT_STATUS.md` for how we obtained it). Download `twcs.csv` from
https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter and place it at:

```
data/raw/twcs.csv
```

Expected: ~516MB, 2,811,774 parsed rows (pandas), columns `tweet_id, author_id, inbound,
created_at, text, response_tweet_id, in_response_to_tweet_id`. The raw file is
**never committed** (`.gitignore`) — see `data/README.md`.

## 6. Reproduce headline results (~8-9 minutes measured, excludes `pip install`)

```bash
bash scripts/run_pipeline.sh
```

This runs, in order: dataset overview + brand-selection evidence (~40s) → conversation
reconstruction + dedup (~80s) → deterministic leakage-safe splits (~5s) → English-language
filtering (~5min, the dominant cost) → taxonomy clustering evidence (~15s, informational
only — the taxonomy itself is already finalized in `configs/intent_taxonomy.json`) →
golden-set (re)build (~5s) → retriever fit + baseline training (~10s) → the evaluation
harness (~5s). Final output: `artifacts/eval_results/{results.json,summary.txt}`.

**Note:** re-running `scripts/05_build_golden_set.py` regenerates
`data/golden/golden_set_provisional.csv` deterministically (fixed seed) but does **not**
overwrite the committed `data/golden/golden_set_reviewed.csv`, which carries the 55
independently-reviewed labels the evaluation harness actually reads
(`configs/eval.yaml: golden_set`). This is intentional — don't lose the review work.

To run just the evaluation harness against already-processed data:
```bash
python -m src.evaluation.run --config configs/eval.yaml
```

To run the test suite (36 tests, no dataset required for most):
```bash
python -m pytest tests/ -v
```

## 7. Headline results (from `artifacts/eval_results/results.json`, n=200, full golden set)

| | Accuracy | Macro-F1 |
|---|---|---|
| Baseline A (majority class) | 6.5% | 1.0% |
| Baseline B (TF-IDF+LogReg) | 37.0% | 42.8% |
| System default (rule classifier, v1.2) | 38.0% | 45.1% |

| Escalation (policy v1.1) | Value |
|---|---|
| Harmful auto-handle rate | 13.5% |
| Unnecessary escalation rate | 30.0% |
| Escalate recall | 72.4% |

Policy v1.1 (`src/escalation/policy.py`) adds a repeat-contact signal and fixes a
profanity-vs-abuse false positive after failure analysis. The n=55→200 golden-set
expansion also surfaced and fixed a real pipeline bug (classification/retrieval were
running on raw, un-normalized tweet text) — see `REPORT.md` §6.4 for the full version-by-
version numbers and §6.5 for a dev/held-out generalization check on policy v1.1.

**Read `REPORT.md` §9 before citing these numbers anywhere** — they're computed on the
full 200-example golden set, all AI-(not independently human-)reviewed, deliberately
rebalanced away from production intent distribution, with no LLM in the loop. They are directional evidence of
a working, measurable system, not a validated production benchmark.

## 8. Example agent output

```json
{
  "intent": "ORDER_STATUS",
  "intent_confidence": 0.75,
  "action": "auto_handle",
  "escalation_reason": null,
  "draft_reply": "Sorry to hear this. Have we missed the delivery date advised? ^PK",
  "evidence": [
    {"pair_id": "AmazonHelp_...", "relevance": 0.87,
     "historical_customer_message": "My Amazon order still hasn't arrived yet...",
     "historical_resolution": "Sorry to hear this. Have we missed the delivery date advised? ^PK"}
  ],
  "response_confidence": 0.87
}
```

## 9. Repository structure

```
README.md  DECISIONS.md  LABELING_GUIDE.md  REPORT.md  BRAND_SELECTION.md
INTENT_TAXONOMY.md  PROJECT_STATUS.md  FINAL_REVIEW.md  LICENSE

configs/           intent_taxonomy.json, eval.yaml
src/data/          reconstruct.py, split.py, lang_filter.py
src/intent/        rule_classifier.py, baselines.py
src/retrieval/     tfidf_retriever.py
src/generation/     responder.py, llm_responder.py
src/escalation/    policy.py
src/evaluation/    run.py, judge_prompt.py, judge_agreement.py
src/pipeline/      agent.py

tests/             36 tests: leakage, escalation, preprocessing, taxonomy, aggregation, malformed input
scripts/           01-06 numbered pipeline steps + run_pipeline.sh
data/golden/       golden_set_reviewed.csv (committed), claude_review_labels.json
data/README.md     dataset instructions
artifacts/         data_quality/ (committed, small), eval_results/ + models/ + index/ (gitignored, regenerable)
```

## 10. Known limitations

English-only (~75% of traffic). Single-label taxonomy. No LLM in reported numbers (code
path exists, unexecuted). 200-example AI-reviewed (not independent human) golden evaluation set. TF-IDF retrieval is lexical, not semantic. 2017-2018 data — AmazonHelp's actual
workflows have changed since. Full unhedged discussion: `REPORT.md` §9-10.

## 11. Reproducibility notes

- All splits are deterministic hash-bucket assignments (`src/data/split.py`), not
  `random.seed`-based — reproducible across machine/Python/numpy versions.
- `src/evaluation/run.py` stamps every result with the git commit hash it was produced at.
- `tests/test_leakage.py` fails the build if golden/retrieval/baseline-training data ever
  overlap.
