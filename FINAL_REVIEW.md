# Final Review — Red-Team Self-Assessment

## Strengths (top 5)
1. **Evidence-based brand selection with a metric that actually matters** — the
   DM-deflection analysis (`BRAND_SELECTION.md`) is the kind of check that distinguishes
   someone who understands *why* grounding requires public resolution text from someone
   who just picked the highest-volume brand.
2. **Leakage safety is enforced by tests, not just documented** — `tests/test_leakage.py`
   fails the build on any golden/retrieval/training overlap; this was caught and fixed
   for real during development (deterministic hash-bucket splits were designed precisely
   to prevent near-duplicate leakage a naive random split would miss).
3. **Honest handling of a real, material blocker (no LLM API key)** — rather than
   fabricating LLM-shaped results or silently degrading scope, the gap is named
   explicitly in three places (`DECISIONS.md` #9, `REPORT.md` §3/§9, this file) with a
   concrete path to close it (`llm_responder.py` and `judge_prompt.py` are fully
   implemented, just unexecuted).
4. **Failure analysis derived from real evaluation output, not invented categories** —
   all 5 failure modes trace to specific example IDs in
   `artifacts/eval_results/results.json`.
5. **Escalation policy is inspectable and tested against its own stated philosophy** —
   7 named signals, OR-logic, and the accepted cost (25.5% unnecessary-escalation rate)
   is reported rather than hidden.

## Biggest weaknesses (top 5)
1. **All 200 golden-set labels used for headline numbers are Claude-reviewed, not
   independently human-verified.** This is the single largest validity gap. It's
   disclosed everywhere, but a reviewer should treat every accuracy/precision/recall
   number as provisional until real human labels exist.
2. **No LLM was actually run.** The system's classification and generation quality
   ceiling is bounded by regex rules and TF-IDF, which is a real (not cosmetic)
   limitation on how good the drafted replies actually are.
3. **Rare-class metrics are still thin even at n=200.** `ABUSE_THREAT_ESCALATION_DEMAND`
   has only 3 golden examples; confusion-matrix cells for rare classes in `results.json`
   should be read as illustrative, not precise.
4. **Baseline B is not independent of the rule classifier** — trained on the rule
   classifier's own silver labels, so the "baseline comparison" is weaker evidence than
   it looks at first glance (disclosed in `REPORT.md` §9 point 5).
5. **English-only scope drops ~25% of real AmazonHelp traffic** — a production system
   built this way would be blind to a meaningful fraction of customers.

## Likely reviewer objections and defenses

**"Your intent classifier is just regex — why should I be impressed?"**
Because the alternative (an LLM) wasn't available in this environment, and rather than
fake it, I built the LLM path for real (`llm_responder.py`, standard Anthropic Messages
API, structured JSON output, system prompt with explicit grounding constraints) and
clearly marked it unexecuted. The regex baseline is honestly evaluated (60% accuracy on
a hard, rebalanced set) rather than dressed up.

**"60% accuracy isn't very good."**
Correct, and I say so — see `REPORT.md` §9 point 3: this is on a set deliberately
rebalanced away from the `OTHER_UNCLEAR`-dominated production distribution, which makes
the number look worse than "accuracy on raw traffic" would, precisely because I refused
to inflate it by letting the golden set be 66% catch-all bucket.

**"Isn't evaluating your rule classifier against labels bootstrapped from that same rule
classifier circular?"**
Yes — which is exactly why I did a second, independent read-through of 55 examples
without consulting the rule classifier's guess (`data/golden/claude_review_labels.json`)
specifically to break that circularity, and restricted the evaluation harness to that
subset (`configs/eval.yaml: only_reviewed_subset: true`).

**"Why TF-IDF and not embeddings?"**
Scale-appropriate (79k documents), fully local, inspectable via literal term overlap, and
the assignment explicitly discourages over-engineered vector infrastructure. Documented
tradeoff: misses true paraphrases with low lexical overlap (`DECISIONS.md` #6).

**"Your escalation policy escalates 25% of cases unnecessarily — isn't that bad?"**
It's a real, measured cost of a policy that's deliberately biased toward escalating
because the assignment states false auto-handling is worse than an unnecessary
escalation. I'd rather report and own that number than tune the policy against the same
55 examples to make it disappear (which would be overfitting, not improvement — see
`REPORT.md` §9 point 8).

**"How do I know the raw dataset wasn't tampered with or substituted?"**
`BRAND_SELECTION.md` records the extraction and schema check
(`md5sum`, row count, column names, date range) at ingestion time; anyone can re-run
`scripts/01_dataset_overview.py` against their own Kaggle download and compare.

**"Why AmazonHelp and not a simpler single-product brand like Spotify?"**
Measured: AmazonHelp had far and away the lowest DM-deflection rate among high-volume
candidates (0.6% vs. 30.8% for Spotify), meaning its public replies actually contain
resolution content to ground on. Spotify would have been an easier taxonomy but a weaker
testbed for the assignment's core claim (grounded-in-historical-resolution).

## Live coding / modification questions (10, with concise answers)

1. **"Add a new intent to the taxonomy — walk me through it."**
   Add an entry to `configs/intent_taxonomy.json` with id/description/inclusion/exclusion/
   examples (tested by `tests/test_taxonomy_schema.py`), add a rule to
   `src/intent/rule_classifier.py`'s `RULES` list at the appropriate priority position,
   retrain Baseline B (`python src/intent/baselines.py`), and add golden examples if you
   want it evaluable.

2. **"How would you change the escalation policy to reduce the unnecessary-escalation
   rate without increasing harmful auto-handles?"**
   Target Failure Analysis #2 specifically: replace the blanket
   `intent in FINANCIAL_INTENTS -> escalate` with a risk-scoped check (e.g., only escalate
   refund requests that mention a dollar amount above a threshold, or a second contact
   about the same order) — implemented as a new named signal, tested against the held-in
   55 examples, then validated it doesn't regress harmful-auto-handle rate before trusting
   it.

3. **"Your retrieval only returns the top-1 match for generation — what if it's wrong?"**
   That's exactly Failure Analysis #5. Two real mitigations: (a) the grounding check
   already blocks concrete unsupported promises regardless of which evidence item was
   used; (b) a similarity-floor fallback to escalation (not currently a distinct signal —
   it's folded into `insufficient_evidence` at threshold 0.3) could be tightened.

4. **"How would you detect near-duplicate customer messages that your exact-hash dedup
   misses?"**
   `content_hash` in `src/data/reconstruct.py` is exact-match only (case/whitespace-
   normalized). A near-duplicate detector would need MinHash/SimHash or embedding-based
   clustering over the customer_text_clean field — not implemented, flagged as a "One
   More Week" gap given the 9.4% observed duplicate rate before exact-dedup.

5. **"Walk me through what happens if `data/raw/twcs.csv` has a different schema next
   year (e.g., a renamed column)."**
   `src/data/reconstruct.py` would raise a `KeyError` on `pd.read_csv` column access —
   there's no schema-validation guard currently. A production version should validate
   expected columns before processing and fail with a clear error rather than a raw
   pandas traceback (not implemented — real gap).

6. **"How do you know the taxonomy clusters weren't just artifacts of TF-IDF
   stopword choices?"**
   I don't with certainty — that's an honest limitation. Mitigation used: cross-checked
   cluster examples by reading actual message text, not just top-terms, before naming
   intents (`INTENT_TAXONOMY.md` step 3), and consolidated clusters that were clearly
   TF-IDF artifacts of overlapping vocabulary (e.g., the three delivery-flavored clusters).

7. **"Show me how you'd add a held-out test to check the escalation policy fix in
   objection #2 above doesn't overfit."**
   Split the 55 reviewed examples further (e.g., 40 dev / 15 held-out) before tuning any
   policy threshold, tune only against dev, report both dev and held-out numbers. Not
   currently done — the 55 examples are used as one set for both failure analysis and
   evaluation, a real methodological gap noted in `REPORT.md` §9 point 8.

8. **"What breaks if two people run `scripts/run_pipeline.sh` on different machines?"**
   Splits and golden-set sampling are deterministic (hash-based / fixed-seed), so
   `data/processed/*` and `data/golden/golden_set_provisional.csv` should be
   bit-for-bit reproducible given the same `twcs.csv`. `data/golden/golden_set_reviewed.csv`
   (the committed, labeled file) is NOT regenerated by the pipeline — it's fixed in git.

9. **"How would you extend this to a second brand?"**
   Rerun `scripts/02`/`03` for evidence, then parameterize `BRAND` in
   `src/data/reconstruct.py` (currently hardcoded), rerun the taxonomy clustering step
   fresh (the taxonomy is brand-specific, not reusable as-is), and build a separate golden
   set. Not a one-line config change today — `BRAND` is a module-level constant, a real
   refactor opportunity.

10. **"Your grounding check is regex — give me an example it would miss."**
    A draft that states a specific wrong fact confidently ("your refund was processed on
    March 3rd") without matching the promise-verb regex pattern would pass the check
    even though it's fabricated — the check only catches the
    promise-verb + absent-from-evidence pattern, not arbitrary factual hallucination.
    Documented in `DECISIONS.md` #12.

## Selection assessment

This submission demonstrates real engineering judgment (evidence-based brand selection,
leakage-safe splits with tests, an escalation policy with a stated philosophy and honestly
reported costs) and real intellectual honesty (the LLM-unavailable gap is the loudest
thing in the report, not hidden in a footnote). The weakest point for a hiring committee
is that the headline numbers rest on a small, AI-reviewed (not human) label set — a
reviewer who reads only the results table without `REPORT.md` §9 could walk away
over-trusting a 60% accuracy figure. The single highest-leverage improvement is getting
real human labels on the golden set (tooling already built,
`scripts/06_review_golden_set.py`) and then re-running the identical evaluation harness —
at that point every other piece of this submission (taxonomy, retrieval, escalation
policy, failure analysis, tests) is already in place to produce a genuinely strong,
defensible result.
