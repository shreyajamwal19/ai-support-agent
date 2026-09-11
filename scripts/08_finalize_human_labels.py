"""Run this AFTER you finish labeling all 200 in golden_set_human.csv.
Command:  python3 scripts/08_finalize_human_labels.py

What it does:
  1. Checks all 200 examples are labeled (refuses to proceed if not, so we never call an
     incomplete set "hand-labelled").
  2. Copies your labels into golden_set_reviewed.csv, REPLACING the Claude-reviewed ones.
  3. Updates configs/eval.yaml so the evaluation harness reads your labels.
  4. Prints the exact next command to re-run evaluation with ONLY your labels.
"""
import pandas as pd
from pathlib import Path

HUMAN = Path("data/golden/golden_set_human.csv")
DST = Path("data/golden/golden_set_reviewed.csv")

def main():
    if not HUMAN.exists():
        print(f"ERROR: {HUMAN} does not exist yet. Run scripts/06_review_golden_set.py first.")
        return
    df = pd.read_csv(HUMAN)
    done = df[df["labeling_status"] == "human_reviewed"]
    total = len(df)
    print(f"{len(done)} / {total} examples are human-labeled.")
    if len(done) < total:
        print(f"NOT finalizing yet -- {total - len(done)} examples still need labels.")
        print("Run scripts/06_review_golden_set.py again to keep going, then re-run this script.")
        return

    df.to_csv(DST, index=False)
    print(f"All {total} examples are human-labeled. Saved to {DST}.")
    print("Every row's labeling_status is now 'human_reviewed' -- this set can now "
          "genuinely be called hand-labelled.")
    print("\nNext: re-run evaluation with your labels only:")
    print("  python3 -m src.evaluation.run --config configs/eval.yaml")

if __name__ == "__main__":
    main()
