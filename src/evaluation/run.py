"""Phase 11: reproducible evaluation harness.
Run: python -m src.evaluation.run --config configs/eval.yaml
Produces machine-readable JSON + human-readable summary + confusion matrix + per-intent
metrics + escalation metrics + failure examples.
"""
from __future__ import annotations
import argparse
import json
import yaml
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                              confusion_matrix, precision_score, recall_score)

from src.intent.rule_classifier import classify as rule_classify
from src.data.reconstruct import clean_text_preserve
from src.intent.baselines import MajorityClassBaseline, TfidfLogRegBaseline
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.escalation.policy import decide as escalation_decide
from src.generation.responder import draft_extractive
from src.llm.provider import get_active_provider_info


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def load_golden(cfg):
    df = pd.read_csv(cfg["golden_set"])
    if cfg.get("only_reviewed_subset", True):
        # Accept either: real human labels (preferred) or the earlier Claude-reviewed
        # bootstrap labels, whichever are present. Once scripts/08_finalize_human_labels.py
        # has run, every row is "human_reviewed" and this naturally uses only those.
        df = df[df["labeling_status"].isin(
            ["human_reviewed", "claude_reviewed_not_independent_human"]
        )].copy()
    return df.reset_index(drop=True)


def eval_intent_classifier(name, preds, gold):
    acc = accuracy_score(gold, preds)
    macro_f1 = f1_score(gold, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(gold, preds, average="weighted", zero_division=0)
    labels = sorted(set(gold) | set(preds))
    p, r, f1, support = precision_recall_fscore_support(gold, preds, labels=labels, zero_division=0)
    per_class = {lab: {"precision": round(float(pp), 3), "recall": round(float(rr), 3),
                        "f1": round(float(ff), 3), "support": int(ss)}
                 for lab, pp, rr, ff, ss in zip(labels, p, r, f1, support)}
    cm = confusion_matrix(gold, preds, labels=labels).tolist()
    return {
        "name": name, "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4), "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class, "confusion_matrix": {"labels": labels, "matrix": cm},
    }


def eval_escalation(preds_action, gold_action):
    labels = ["auto_handle", "escalate"]
    p, r, f1, support = precision_recall_fscore_support(gold_action, preds_action, labels=labels, zero_division=0)
    # harmful auto-handle rate: gold says escalate but system said auto_handle
    n = len(gold_action)
    harmful_auto = sum(1 for g, pr in zip(gold_action, preds_action) if g == "escalate" and pr == "auto_handle")
    unnecessary_esc = sum(1 for g, pr in zip(gold_action, preds_action) if g == "auto_handle" and pr == "escalate")
    return {
        "escalate_precision": round(float(p[1]), 3), "escalate_recall": round(float(r[1]), 3),
        "escalate_f1": round(float(f1[1]), 3),
        "auto_handle_precision": round(float(p[0]), 3), "auto_handle_recall": round(float(r[0]), 3),
        "n_examples": n,
        "harmful_auto_handle_count": harmful_auto,
        "harmful_auto_handle_rate": round(harmful_auto / n, 4) if n else None,
        "unnecessary_escalation_count": unnecessary_esc,
        "unnecessary_escalation_rate": round(unnecessary_esc / n, 4) if n else None,
    }


def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    golden = load_golden(cfg)
    print(f"Evaluating on {len(golden)} examples (only_reviewed_subset={cfg.get('only_reviewed_subset')})")

    exclude_ids = set(pd.read_csv(cfg["golden_set"])["pair_id"])  # exclude ALL golden pairs from retrieval, not just reviewed
    retriever = TfidfRetriever().load()

    texts = golden["customer_text_raw"].fillna("").tolist()
    # classify/retrieve on cleaned text -- same normalization the retrieval index and
    # rule regexes were built against (see src/pipeline/agent.py bug-fix note, DECISIONS.md #17)
    clean_texts = [clean_text_preserve(t) for t in texts]
    gold_intent = golden["gold_intent"].tolist()
    gold_action = golden["gold_action"].tolist()

    # --- Intent: Baseline A (majority class, fit on baseline_eval_pool silver labels) ---
    train_pool = pd.read_parquet("data/processed/baseline_eval_pool_en.parquet")
    silver = train_pool["customer_text_clean"].apply(rule_classify).apply(lambda d: d["intent"])
    baseline_a = MajorityClassBaseline().fit(silver)
    preds_a = baseline_a.predict(clean_texts)

    # --- Intent: rule classifier (production default) ---
    rule_preds = [rule_classify(t)["intent"] for t in clean_texts]
    rule_confs = [rule_classify(t)["confidence"] for t in clean_texts]

    # --- Intent: Baseline B (TF-IDF + LogReg) ---
    baseline_b = TfidfLogRegBaseline().load()
    preds_b = list(baseline_b.predict(clean_texts))

    intent_results = [
        eval_intent_classifier("baseline_A_majority_class", preds_a, gold_intent),
        eval_intent_classifier("baseline_B_tfidf_logreg", preds_b, gold_intent),
        eval_intent_classifier("system_rule_classifier", rule_preds, gold_intent),
    ]

    # --- Retrieval + escalation + generation, run through the actual pipeline ---
    from src.pipeline.agent import SupportAgent
    agent = SupportAgent(retriever=retriever)
    pipeline_outputs = []
    for t in texts:
        out = agent.handle(t, exclude_pair_ids=exclude_ids)  # agent cleans internally
        pipeline_outputs.append(out)

    preds_action = [o["action"] for o in pipeline_outputs]
    escalation_results = eval_escalation(preds_action, gold_action)

    grounding_pass_rate = float(np.mean([
        draft_extractive(t, o["evidence"], o["intent"])["grounding"]["passed"]
        for t, o in zip(texts, pipeline_outputs)
    ]))

    retrieval_hit_rate = float(np.mean([len(o["evidence"]) > 0 for o in pipeline_outputs]))
    avg_top1_relevance = float(np.mean([o["response_confidence"] for o in pipeline_outputs]))

    # --- Failure collection: cases where system action != gold action ---
    failures = []
    for i, (t, o, ga) in enumerate(zip(texts, pipeline_outputs, gold_action)):
        if o["action"] != ga:
            failures.append({
                "example_id": golden.iloc[i]["example_id"], "text": t,
                "gold_intent": gold_intent[i], "system_intent": o["intent"],
                "gold_action": ga, "system_action": o["action"],
                "system_escalation_reason": o["escalation_reason"],
            })

    results = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(),
            "config": cfg,
            "n_golden_examples_used": len(golden),
            "llm_provider": get_active_provider_info(),
            "caveat": (f"Metrics below are computed against {len(golden)} examples with "
                       "labeling_status in {human_reviewed, claude_reviewed_not_independent_human}. "
                       "Unless labeling_status is 'human_reviewed' for all of them, these are "
                       "NOT independently human-verified labels -- treat as directional, not "
                       "final validated numbers. See REPORT.md 'What is misleading about my "
                       "headline number'. llm_provider above reflects the LLM_PROVIDER "
                       "configured for this run even though the classical (rule/TF-IDF) path "
                       "is what actually produced the numbers below -- see DECISIONS.md."),
        },
        "intent_classification": intent_results,
        "escalation": escalation_results,
        "retrieval": {"hit_rate_at_min_similarity_0.15": round(retrieval_hit_rate, 4),
                       "avg_top1_relevance": round(avg_top1_relevance, 4)},
        "generation_grounding_pass_rate": round(grounding_pass_rate, 4),
        "action_disagreement_failures": failures,
    }

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # human-readable summary
    lines = []
    lines.append(f"=== Evaluation Summary ({results['metadata']['timestamp']}) ===")
    lines.append(f"git commit: {results['metadata']['git_commit'][:8]}")
    lines.append(f"N golden examples (reviewed subset): {len(golden)}\n")
    lines.append("-- Intent classification (accuracy / macro-F1) --")
    for r in intent_results:
        lines.append(f"  {r['name']:30s} acc={r['accuracy']:.3f}  macro_f1={r['macro_f1']:.3f}  weighted_f1={r['weighted_f1']:.3f}")
    lines.append("\n-- Escalation --")
    for k, v in escalation_results.items():
        lines.append(f"  {k}: {v}")
    lines.append("\n-- Retrieval / Grounding --")
    lines.append(f"  retrieval hit-rate (>=0.15 sim): {retrieval_hit_rate:.3f}")
    lines.append(f"  avg top-1 relevance: {avg_top1_relevance:.3f}")
    lines.append(f"  generation grounding-check pass rate: {grounding_pass_rate:.3f}")
    lines.append(f"\n-- Action disagreement failures: {len(failures)} / {len(golden)} --")
    summary = "\n".join(lines)
    with open(out_dir / "summary.txt", "w") as f:
        f.write(summary)
    print(summary)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    args = ap.parse_args()
    main(args.config)
