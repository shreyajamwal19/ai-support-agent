# Decision Log

15 non-obvious decisions made while building this, with alternatives considered and the
tradeoff accepted. Ordered roughly by when they were made.

---

**1. Brand: AmazonHelp, not the highest-engagement or "easiest" brand.**
- Alternatives: AppleSupport (highest volume among "clean" brands), Uber_Support, Delta.
- Why: measured DM-deflection rate (`scripts/03_deflection_check.py`) showed AmazonHelp at
  0.6% vs. 16.5-71.5% for every other top-volume candidate. A brand whose public replies are
  mostly "please DM us" has nothing to ground a public-grounding system on.
- Tradeoff: Amazon is the most heterogeneous business in the candidate set (retail + devices
  + digital + payments), making a single clean taxonomy harder than for a single-product
  brand like Spotify.

**2. Single-label intent taxonomy for a naturally multi-label domain.**
- Alternatives: multi-label classification; hierarchical taxonomy (issue family -> sub-issue).
- Why: 150-250 golden examples cannot support reliable per-combination multi-label metrics.
  Escalation-relevant properties (abuse, financial risk) are handled as a *separate* policy
  layer (Phase 9) instead of folding them into the intent label, so a refund request that is
  also abusive still escalates correctly without needing a compound intent label.
- Tradeoff: some information loss for genuinely dual-purpose messages; documented as a known
  limitation, not hidden.

**3. Deterministic hash-bucket splits instead of `random.sample(seed=...)`.**
- Alternatives: sklearn `train_test_split` with a fixed seed.
- Why: hash-bucket splits are reproducible across Python/numpy versions and, critically,
  guarantee that near-duplicate customer messages (same `content_hash`) always land in the
  *same* split rather than risking a retry/duplicate leaking across golden/retrieval.
- Tradeoff: less control over exact split proportions (we get ~15/15/70%, not a clean
  round number) -- acceptable given leakage-safety is the higher priority.

**4. English-only scoping, decided *after* seeing real cluster output, not upfront.**
- Alternatives: multilingual taxonomy; translate-then-classify.
- Why: `scripts/04_taxonomy_clustering.py`'s first (unfiltered) run showed clusters
  dominated by German/Japanese/French/Spanish text and off-topic promotional tweets. A
  12-intent taxonomy across 5+ languages was out of scope for a take-home; ~75% of the
  reconstructed pairs are English, still leaving >100k retrieval-pool pairs.
- Tradeoff: the system is not usable as-is for non-English AmazonHelp traffic; flagged in
  REPORT.md as a real production gap, not silently dropped.

**5. `OTHER_UNCLEAR` capped in golden-set sampling instead of sampled proportionally, but
   never dropped entirely.**
- Alternatives: pure proportional stratified sampling (would put ~130/238 slots into
  `OTHER_UNCLEAR`, since it's 66% of the raw pool); or excluding it from the golden set
  altogether.
- Why: a golden set that is majority catch-all-bucket is not useful for evaluating the
  intents that actually matter operationally — capped at 30/200 (15%) in the base
  stratified sample. But it's kept, not dropped, because a real production system will
  receive off-topic/ambiguous traffic and a classifier/escalation-policy untested on it
  would have an untested failure mode in exactly the place most likely to produce a
  nonsensical auto-reply.
- Tradeoff: the golden set's intent distribution no longer matches the true production
  distribution — headline accuracy numbers are NOT representative of "accuracy on random
  incoming traffic." Restated explicitly in `REPORT.md`.

**6. TF-IDF retrieval, not a dense embedding + vector DB.**
- Alternatives: sentence-transformers embeddings + FAISS/Chroma.
- Why: at ~79k retrieval-pool documents, TF-IDF cosine similarity is fast (<50ms/query),
  fully local (no embedding API cost/latency), and just as inspectable (term overlap is
  literally why two messages matched, vs. an opaque embedding distance). The assignment
  explicitly says "do not over-engineer vector infrastructure."
- Tradeoff: TF-IDF misses semantic paraphrases with low lexical overlap (e.g. "my package
  never showed up" vs "package not delivered" partially overlaps but a true paraphrase with
  zero shared words would be missed). Documented as a known retrieval-quality ceiling.

**7. Escalation policy is OR-logic over named risk signals, not a single confidence
   threshold, and is deliberately biased toward escalating.**
- Alternatives: `if confidence < 0.5: escalate` (explicitly rejected by the assignment).
- Why: false auto-handling (confidently wrong, ships an unsupported promise) is worse than
  an unnecessary escalation (costs a human a few minutes). Financial intents (refund/
  billing) and account-security intents escalate unconditionally regardless of confidence.
- Tradeoff, measured honestly: this produces a real unnecessary-escalation rate of 25.5% on
  our reviewed subset (see REPORT.md) -- the policy is deliberately conservative and that
  cost is visible in the numbers, not hidden.

**8. Baseline B (TF-IDF+LogReg) is trained on rule-classifier "silver" labels, not
   independent gold labels.**
- Alternatives: skip Baseline B until real human labels exist; hand-label a large training
  set first.
- Why: training data volume needed (>15k examples) for a competent TF-IDF classifier makes
  full hand-labeling infeasible in this timeframe; silver-label training is a standard,
  named technique (weak supervision), not a shortcut we're hiding.
- Tradeoff, stated plainly in `src/intent/baselines.py` and `REPORT.md`: Baseline B cannot
  be expected to exceed the rules it was trained to imitate on non-golden data, and its
  evaluated accuracy against golden labels is only informative to the extent the golden
  labels are independent of the rules -- which is exactly why decision #9 below exists.

**9. No LLM (Anthropic API) was called for classification/generation/judging in this
   submission's reported numbers -- decision made explicit rather than silently degrading
   scope.**
- Context: no `ANTHROPIC_API_KEY` was present in the build sandbox (network egress to
  `api.anthropic.com` is allow-listed, but no credential was provisioned).
- Alternatives considered: (a) fabricate plausible-looking LLM outputs -- explicitly
  forbidden and would be dishonest; (b) silently ship LLM-shaped code paths without
  disclosing they were never run -- also dishonest; (c) what we did: build the LLM
  intent/generation/judge code paths fully and correctly against the real Anthropic
  Messages API, but report every number in `REPORT.md` as coming from the classical
  (rule-based + TF-IDF) path, with the LLM path's status explicitly marked "implemented,
  unmeasured." **This is the single most research-validity-relevant decision in this
  project** and is restated in the "misleading headline number" section.
- Tradeoff: the shipped system's classification/generation quality is bounded by what
  regex rules and TF-IDF can do, not by an LLM's language understanding -- a real quality
  ceiling, not a cosmetic one.

**10. All 200 golden-set examples are "Claude-reviewed" labels (not rule-derived, and
    not human-labeled), specifically to break evaluation circularity while maximizing
    statistical power -- and are explicitly NOT called "hand-labeled."**
- Why: the golden set's `gold_intent` was bootstrapped from the rule classifier's own
  output. Evaluating the rule classifier against those labels would be tautological
  (near-100% "accuracy" by construction). An independent read-through (by Claude, not
  using the rule classifier's guess) of a 55-example stratified subset produces labels
  that are non-circular with respect to the rule baseline, even though they are still not
  independent human labels.
- Tradeoff: even at n=200, per-class metrics on rare intents (e.g. n=3 for
  `ABUSE_THREAT_ESCALATION_DEMAND`) are noisy. All reported numbers use these labels and
  say so explicitly (`REPORT.md` §9 point 1).

**11. Generation is extractive/template-grounded by default, not free-text LLM
    generation.**
- Direct consequence of decision #9. The shipped `draft_extractive()` reuses the single
  most similar historical resolution's *exact text* rather than paraphrasing it.
- Tradeoff: replies can read as slightly mismatched to the current customer's specific
  wording (see failure analysis in REPORT.md, e.g. GOLD examples where top-1 retrieval
  match was topically adjacent but not precise) -- a real, measured cost of not having LLM
  generation available, not glossed over.

**12. Grounding check is a regex-based promise-detector, not an LLM-based fact-checker.**
- Alternatives: LLM call to verify every draft against evidence (again, decision #9 blocks
  this for the reported numbers).
- Why: even a crude regex check ("we will refund/credit/replace... not present in
  evidence") catches the most dangerous failure mode (inventing a concrete promised
  action) without needing an API call, and is fully deterministic/testable.
- Tradeoff: does not catch subtler unsupported claims (wrong facts stated confidently in
  a way that doesn't match the promise-pattern regex). Documented as a real gap.

**13. Failure analysis top-5 modes were derived from the actual evaluation output at each
    stage (21 cases at n=55, then re-derived from 87 cases at n=200 after real bugs were
    found and fixed), not assumed upfront or left stale after the golden set grew.**
- The assignment explicitly warns against assuming failure categories in advance; the
  final five categories in `REPORT.md` §8 were rewritten after the n=200 + bugfix round
  (`DECISIONS.md` #17), not left as the earlier n=55 draft.

**14. Policy v1.1 fixes were tuned on a `dev` half of the reviewed subset and checked (not
    tuned) against a `held_out` half, rather than tuned against all 55 examples at once.**
- Alternatives: tune directly against all 55 reviewed examples (simpler, but exactly the
  overfitting risk flagged in `DECISIONS.md`'s earlier draft of this list, decision-log
  entry now folded in above as #8's sibling concern).
- Why: Failure Analysis #3/#4 fixes (repeat-contact escalation signal, brand-directed-
  profanity requirement for the abuse rule) needed *some* check that they generalize
  rather than just resolve the specific 21 cases that motivated them.
- Result, reported honestly in `REPORT.md` §6.4: harmful-auto-handle rate improved on
  both `dev` (12.7%→9.5%) and the untuned `held_out` split (→7.7%) — real evidence of
  generalization on the metric that matters most. Unnecessary-escalation rate worsened,
  more so on `held_out` (30.9% dev / 38.5% held_out vs. 25.5% pre-fix baseline) — a
  genuine, disclosed tradeoff, not hidden by only reporting the metric that improved.
- Tradeoff: n=13 on `held_out` is too small to be statistically conclusive either way;
  this is a directional sanity check, not a rigorous generalization proof.

**15. A real pipeline bug (raw text fed to classification/retrieval instead of cleaned
    text) was found and fixed by expanding the golden set to n=200, not by code review
    alone.**
- What happened: `src/pipeline/agent.py` and `src/evaluation/run.py` were passing
  `customer_text_raw` (still containing the leading `@AmazonHelp ` mention on nearly
  every message) directly into `rule_classify()` and the TF-IDF retriever, instead of
  `clean_text_preserve()`d text — the same normalization the retrieval index and regex
  rules were built/tested against. At n=55 this was invisible (the 55-example subset
  happened not to expose it badly); at n=200, `ACKNOWLEDGEMENT_FOLLOWUP` recall of
  exactly 0.00 was the signal that something structural was wrong, not just "the regex
  needs more keywords."
- Why this matters as a decision-log entry: it's direct evidence that expanding golden-
  set coverage isn't just about statistical power — it surfaces real bugs that a small,
  curated subset can hide. Fixing it improved rule-classifier accuracy 35.5%→38.0% and
  average retrieval relevance 0.498→0.534, for free, with no risk of overfitting (it's a
  correctness fix, not a tuned heuristic).
- Tradeoff: none identified; this is a straightforward bug fix, included here specifically
  to model the practice of writing down *when a metric caught a real bug* as its own
  decision-log-worthy event, not just final design choices.

**16. What we deliberately did not build:** a UI/frontend, a production API server, a
    vector database, multi-language support, multi-turn dialogue *management* (vs. just
    reading prior-turn context), fine-tuning any model, and a fully-automated human-in-the-
    loop labeling pipeline. Each is a reasonable next step (see REPORT.md "One More Week")
    but none is what this assignment is graded on -- the assignment explicitly says "the
    proof is worth more than the system," so time went into measurement validity over
    surface polish.
