"""Minimal terminal CLI for a human to review/correct the provisional golden set.
Run: python3 scripts/06_review_golden_set.py
Writes incrementally to data/golden/golden_set_reviewed.csv so it's safe to stop/resume.
"""
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timezone

SRC = Path("data/golden/golden_set_provisional.csv")
DST = Path("data/golden/golden_set_reviewed.csv")
TAXONOMY = json.load(open("configs/intent_taxonomy.json"))["intents"]
INTENT_IDS = [i["id"] for i in TAXONOMY]

def load():
    if DST.exists():
        return pd.read_csv(DST)
    df = pd.read_csv(SRC)
    return df

def main():
    df = load()
    todo = df[df["labeling_status"] != "human_reviewed"]
    print(f"{len(todo)} examples remaining to review (of {len(df)} total)")
    annotator = input("Annotator initials: ").strip() or "anon"
    for idx, row in todo.iterrows():
        print("\n" + "="*70)
        print(f"[{row['example_id']}] {row['customer_text_raw']}")
        if row.get("n_prior_context_turns", 0):
            print(f"  (has {row['n_prior_context_turns']} prior context turns -- see prior_context column)")
        print(f"  provisional_intent (rule baseline guess, for reference only): {row['provisional_intent']}")
        print(f"  Intents: {', '.join(INTENT_IDS)}")
        intent = input("  gold_intent> ").strip().upper() or row["provisional_intent"]
        action = input("  gold_action [auto_handle/escalate]> ").strip() or "auto_handle"
        rationale = ""
        if action == "escalate":
            rationale = input("  gold_escalation_rationale> ").strip()
        conf = input("  label_confidence [high/medium/low]> ").strip() or "medium"
        notes = input("  annotator_notes (optional)> ").strip()

        df.loc[idx, "gold_intent"] = intent
        df.loc[idx, "gold_action"] = action
        df.loc[idx, "gold_escalation_rationale"] = rationale
        df.loc[idx, "label_confidence"] = conf
        df.loc[idx, "annotator_notes"] = notes
        df.loc[idx, "labeling_status"] = "human_reviewed"
        df.loc[idx, "annotator"] = annotator
        df.loc[idx, "reviewed_at"] = datetime.now(timezone.utc).isoformat()

        df.to_csv(DST, index=False)  # incremental save

        cont = input("Continue? [Y/n]> ").strip().lower()
        if cont == "n":
            break
    print(f"Saved progress to {DST}")

if __name__ == "__main__":
    main()
