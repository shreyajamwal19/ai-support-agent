"""
SIMPLE HAND-LABELING TOOL -- no coding knowledge needed.
Run with:  python3 scripts/06_review_golden_set.py

For each of 200 customer messages, you will:
  1. Read the message.
  2. Type a NUMBER to pick the intent (a menu is shown every time).
  3. Type a NUMBER to say auto_handle or escalate.
  4. If you pick escalate, type a NUMBER for why.
  5. Press Enter to save and move to the next one.

You can stop anytime (Ctrl+C or answer 'q') and resume later -- your progress is saved
after every single example, nothing is lost.
"""
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timezone

SRC = Path("data/golden/golden_set_reviewed.csv")   # current (Claude-reviewed) labels
DST = Path("data/golden/golden_set_human.csv")        # YOUR labels go here, separate file
TAXONOMY = json.load(open("configs/intent_taxonomy.json"))["intents"]

RATIONALES = [
    "insufficient_evidence", "financial_refund", "account_security",
    "abuse_legal_threat", "novel_unsupported", "low_intent_confidence", "other",
]

INTRO = """
========================================================================
 HAND-LABELING TOOL -- read this once before you start
========================================================================
WHAT YOU ARE LOOKING AT:
  A real customer tweet sent to @AmazonHelp (Amazon's Twitter support account).
  Sometimes you'll also see "context" -- earlier messages in the same
  conversation, shown oldest first, so you understand what the customer means.

WHAT THE INTENTS MEAN:
  Each message needs ONE label describing what the customer wants. The menu
  below shows all 12 options with a one-line description every single time,
  so you never have to memorize anything.

HOW TO CHOOSE AN INTENT:
  Read the message. Ask "what is this person actually asking for?" Pick the
  option that matches their MAIN request. If two seem close, pick whichever
  one's description matches best -- there's a spot to leave a note if you're
  unsure, and unsure is a totally fine answer (there's a confidence field).

HOW TO CHOOSE auto_handle vs escalate:
  Ask yourself: "Could a scripted/automatic reply safely handle this, or does
  a real human need to look at this person's account/order/case?"
  - auto_handle = a generic, safe reply is fine (status update, simple info,
    a thank-you, a common question).
  - escalate = money, refunds, account security/login, anger/legal threats,
    or anything where a wrong automatic reply could make things worse.
  When in doubt, choose escalate -- it's the safer mistake.

HOW MANY EXAMPLES:
  200 total. The tool tells you how many are left every time you start it.
  Budget about 20-30 seconds per example (~60-90 minutes total). You can do
  them in multiple sittings.

HOW YOUR ANSWERS ARE SAVED:
  Every single answer is saved immediately to a new file:
    data/golden/golden_set_human.csv
  This file does NOT touch the old Claude-reviewed labels -- it's separate,
  so nothing you had before is at risk. Closing the terminal is always safe.
========================================================================
"""

def load():
    if DST.exists():
        df = pd.read_csv(DST)
    else:
        df = pd.read_csv(SRC)
        df["labeling_status"] = "not_yet_human_labeled"
    for col in ["gold_intent", "gold_action", "gold_escalation_rationale",
                "label_confidence", "annotator_notes", "labeling_status",
                "annotator", "reviewed_at"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(object)
    return df

def show_intent_menu():
    print("\n  Pick the intent:")
    for i, intent in enumerate(TAXONOMY, 1):
        print(f"   {i:2d}. {intent['id']:32s} - {intent['description']}")

def ask_number(prompt, n_options):
    while True:
        raw = input(prompt).strip().lower()
        if raw == "q":
            return None
        if raw.isdigit() and 1 <= int(raw) <= n_options:
            return int(raw)
        print(f"  Please type a number from 1 to {n_options} (or 'q' to stop for now).")

def main():
    print(INTRO)
    df = load()
    todo = df[df["labeling_status"] != "human_reviewed"]
    print(f"You have {len(todo)} of {len(df)} examples left to review.\n")
    if len(todo) == 0:
        print("All 200 are already labeled! Nothing left to do.")
        return
    annotator = input("Type your name/initials, then press Enter: ").strip() or "reviewer"

    for idx, row in todo.iterrows():
        print("\n" + "=" * 72)
        print(f"[{row['example_id']}]")
        if row.get("n_prior_context_turns", 0):
            try:
                ctx = json.loads(row["prior_context"].replace("'", '"')) if isinstance(row.get("prior_context"), str) else row.get("prior_context")
            except Exception:
                ctx = None
            if ctx:
                print("  --- earlier context (oldest first) ---")
                for turn in ctx:
                    speaker = "CUSTOMER" if turn.get("inbound") else "AMAZON"
                    print(f"    {speaker}: {turn.get('text','')}")
                print("  --- end context ---")
        print(f"\n  CUSTOMER MESSAGE:\n  \"{row['customer_text_raw']}\"\n")

        show_intent_menu()
        choice = ask_number("  Your choice (number, or 'q' to stop): ", len(TAXONOMY))
        if choice is None:
            break
        intent = TAXONOMY[choice - 1]["id"]

        print("\n  Should this be handled automatically, or sent to a human?")
        print("   1. auto_handle  (a safe generic reply is fine)")
        print("   2. escalate     (a human should look at this)")
        action_choice = ask_number("  Your choice (1 or 2, or 'q' to stop): ", 2)
        if action_choice is None:
            break
        action = "auto_handle" if action_choice == 1 else "escalate"

        rationale = ""
        if action == "escalate":
            print("\n  Why escalate? (pick the closest reason)")
            for i, r in enumerate(RATIONALES, 1):
                print(f"   {i}. {r}")
            r_choice = ask_number("  Your choice (or 'q' to stop): ", len(RATIONALES))
            if r_choice is None:
                break
            rationale = RATIONALES[r_choice - 1]

        print("\n  How confident are you in this label?")
        print("   1. high   2. medium   3. low")
        c_choice = ask_number("  Your choice (or 'q' to stop): ", 3)
        if c_choice is None:
            break
        confidence = ["high", "medium", "low"][c_choice - 1]

        notes = input("  Optional note (press Enter to skip): ").strip()

        df.loc[idx, "gold_intent"] = intent
        df.loc[idx, "gold_action"] = action
        df.loc[idx, "gold_escalation_rationale"] = rationale
        df.loc[idx, "label_confidence"] = confidence
        df.loc[idx, "annotator_notes"] = notes
        df.loc[idx, "labeling_status"] = "human_reviewed"
        df.loc[idx, "annotator"] = annotator
        df.loc[idx, "reviewed_at"] = datetime.now(timezone.utc).isoformat()
        df.to_csv(DST, index=False)  # saved immediately, every example

        remaining = (df["labeling_status"] != "human_reviewed").sum()
        print(f"  Saved. {remaining} left.")

    print(f"\nProgress saved to {DST}. Run this script again anytime to continue.")

if __name__ == "__main__":
    main()
