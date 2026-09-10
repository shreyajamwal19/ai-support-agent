# Intent Taxonomy

Structured, version-controlled data: `configs/intent_taxonomy.json` (12 intents, including
`OTHER_UNCLEAR`). This file is the single source of truth consumed by classifier code,
labeling tooling, and the report.

## How it was derived
1. `scripts/04_taxonomy_clustering.py` ran TF-IDF + MiniBatchKMeans (k=12) over the English-
   filtered `baseline_eval_pool` customer messages (independent of the golden set, no
   leakage). Raw cluster output: `artifacts/data_quality/taxonomy_clusters_en.json`.
2. Clusters were inspected by top TF-IDF terms + 5 example messages each.
3. Clusters were manually named, merged, or split based on whether they represented a
   distinct *actionable* customer intent (e.g. the three delivery-flavored clusters were
   consolidated into `ORDER_STATUS` vs `DELIVERY_QUALITY_ISSUE`, a real and actionable
   distinction, rather than kept as three separate near-duplicate intents).
4. `ACKNOWLEDGEMENT_FOLLOWUP` and `ABUSE_THREAT_ESCALATION_DEMAND` were added by hand: the
   former because short dialogue-act replies ("Yes", "Details sent.") appeared throughout
   clusters as noise rather than their own coherent cluster; the latter because escalation-
   relevant hostile/legal-threat language is safety-critical (feeds Phase 9 escalation
   policy directly) even though it wasn't a numerically dominant cluster.
5. `OTHER_UNCLEAR` is the explicit catch-all for off-topic/ambiguous text (the clustering
   pass showed real off-topic volume: promotional mentions, unrelated commentary).

## Known limitation
This is a **single-label** taxonomy for a naturally multi-label domain (a message can be
both `ABUSE_THREAT_ESCALATION_DEMAND` and `REFUND_RETURN`). We chose single-label because
(a) the golden-set size (150-250) can't support reliable multi-label metrics per combination,
and (b) the escalation policy (Phase 9) is evaluated as a *separate* task from intent, so
abusive-refund-requests still get flagged for escalation even if intent-labeled as
`REFUND_RETURN`. See `DECISIONS.md`.
