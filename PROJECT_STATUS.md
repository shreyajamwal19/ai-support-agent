# Project Status

**Status: submission-ready, with honestly disclosed limitations (see `REPORT.md` §9 and
`FINAL_REVIEW.md`).**

## Completed
- [x] Dataset verified (twcs.csv, md5 `73e961b2837626de89618a3f35f7bd6c`, 2,811,774 parsed rows)
- [x] Evidence-based brand selection (AmazonHelp) — `BRAND_SELECTION.md`
- [x] Preprocessing: conversation reconstruction, dedup, English-language filtering,
      deterministic leakage-safe splits — `src/data/`
- [x] Intent taxonomy (12 classes) derived from TF-IDF+KMeans clustering evidence —
      `INTENT_TAXONOMY.md`, `configs/intent_taxonomy.json`
- [x] Golden evaluation set: 200 examples, stratified sampling, all 200 independently
      (Claude-)reviewed to break evaluation circularity, tooling for full human review —
      `LABELING_GUIDE.md`, `scripts/06_review_golden_set.py`
- [x] Baselines A (majority class) and B (TF-IDF+LogReg) — `src/intent/baselines.py`
- [x] Retrieval: TF-IDF cosine similarity over 78,624-pair English retrieval pool —
      `src/retrieval/tfidf_retriever.py`
- [x] Escalation policy: 7 named risk signals, OR-logic — `src/escalation/policy.py`
- [x] Generation: extractive/template-grounded + regex grounding check —
      `src/generation/responder.py`; LLM path implemented, unexecuted —
      `src/generation/llm_responder.py`
- [x] Evaluation harness: single command, JSON + human-readable output —
      `src/evaluation/run.py`
- [x] LLM-judge rubric + agreement statistics: implemented, unexecuted (no API key) —
      `src/evaluation/judge_prompt.py`, `judge_agreement.py`
- [x] Failure analysis: 5 modes derived from real evaluation output — `REPORT.md` §8
- [x] Report (all required sections) — `REPORT.md`
- [x] Decision log (15 decisions) — `DECISIONS.md`
- [x] Test suite: 36 tests, all passing — `tests/`
- [x] Escalation policy v1.1/v1.2 + a real pipeline text-normalization bug found and
      fixed while expanding the golden set to n=200 — `DECISIONS.md` #14/#15
- [x] README with <15-min reproduction path (measured ~8-9 min) — `README.md`
- [x] Final red-team review — `FINAL_REVIEW.md`

## Known, disclosed gaps (not attempted to hide)
- Golden-set labels are AI-reviewed (Claude, this session) for all 200/200 examples, not
  independently human-verified by the assignment author.
- LLM path (classification/generation/judge) implemented but never executed — no
  `ANTHROPIC_API_KEY` in the build sandbox.
- English-only scope (~75% of reconstructed traffic).
- Single-label taxonomy for a naturally multi-label domain.

## Environment notes for reproduction
- Kaggle's API/website was unreachable from the build sandbox's network egress proxy
  (`host_not_allowed` on `kaggle.com`, `www.kaggle.com`, `api.kaggle.com`,
  `storage.googleapis.com` — confirmed via direct requests and the official `kaggle` CLI).
  The dataset was obtained via direct upload instead. Anyone reproducing this on a machine
  with normal internet access can use the Kaggle CLI/website directly.
- GitHub push access required a user-provided fine-grained PAT (not stored in this repo;
  configured via a local git credential helper for this session only).
