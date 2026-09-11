"""Evaluate the v1.1 policy/rule refinements (repeat-contact signal, brand-directed-
profanity requirement for abuse) on dev vs held-out halves of the 55-example reviewed
subset, to check whether fixes tuned on `dev` generalize to `held_out` rather than just
overfitting to the failure cases that motivated them. See REPORT.md Failure Analysis
#3/#4 and DECISIONS.md #16.
"""
import sys
sys.path.insert(0, ".")
import pandas as pd
import json
from src.pipeline.agent import SupportAgent
from src.retrieval.tfidf_retriever import TfidfRetriever

df = pd.read_csv("data/golden/reviewed_dev_heldout_split.csv")
exclude_ids = set(pd.read_csv("data/golden/golden_set_reviewed.csv")["pair_id"])

retriever = TfidfRetriever().load()
agent = SupportAgent(retriever=retriever)

results = {}
for split_name in ["dev", "held_out"]:
    sub = df[df["eval_split"] == split_name]
    n = len(sub)
    harmful_auto = 0
    unnecessary_esc = 0
    correct = 0
    for _, row in sub.iterrows():
        out = agent.handle(row["customer_text_raw"], exclude_pair_ids=exclude_ids)
        gold = row["gold_action"]
        sysA = out["action"]
        if gold == sysA:
            correct += 1
        if gold == "escalate" and sysA == "auto_handle":
            harmful_auto += 1
        if gold == "auto_handle" and sysA == "escalate":
            unnecessary_esc += 1
    results[split_name] = {
        "n": n, "action_accuracy": round(correct / n, 3),
        "harmful_auto_handle_rate": round(harmful_auto / n, 3),
        "unnecessary_escalation_rate": round(unnecessary_esc / n, 3),
    }

print(json.dumps(results, indent=2))
with open("artifacts/eval_results/policy_v1_1_dev_heldout.json", "w") as f:
    json.dump(results, f, indent=2)
