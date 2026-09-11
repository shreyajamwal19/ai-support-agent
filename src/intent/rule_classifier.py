"""Rule-based intent classifier (Baseline B component + fallback signal source).
Keyword/regex rules derived directly from the taxonomy's inclusion_criteria and the
cluster top-terms. This is intentionally simple and fully inspectable -- every decision
traces to an explicit rule, no black box.
"""
import json
import re
from pathlib import Path

TAXONOMY_PATH = Path("configs/intent_taxonomy.json")

with open(TAXONOMY_PATH) as f:
    TAXONOMY = json.load(f)
INTENT_IDS = [i["id"] for i in TAXONOMY["intents"]]

# Ordered rules: first match wins. Order encodes priority (e.g. abuse/escalation checked
# before generic complaint; refund checked before generic order status).
# v1.1: bare profanity used as an intensifier ("the piece of shit doesn't work") should
# not alone trigger ABUSE_THREAT_ESCALATION_DEMAND -- require it to be brand/staff-
# directed, OR a legal/explicit-escalation-demand phrase (which needs no proximity check).
# Fixes Failure Analysis #4, tuned on the `dev` half of the reviewed subset only.
_ABUSE_LEGAL_OR_DEMAND = re.compile(
    r"\b(lawyer|sue|lawsuit|fraud|bbb|attorney general|speak to (a )?manager|corporate|"
    r"escalat\w*|legal action)\b", re.I)
_ABUSE_BRAND_DIRECTED_PROFANITY = re.compile(
    r"\b(you|your|yall|y'all|customer service|support|rep(s)?|agent(s)?)\b.{0,40}"
    r"\b(fuck|shit|useless|worthless|pathetic|incompetent|idiot|garbage)\b"
    r"|\b(fuck|shit|useless|worthless|pathetic|incompetent|idiot|garbage)\b.{0,40}"
    r"\b(you|your|customer service|support|rep(s)?|agent(s)?)\b", re.I)


class _AbuseRule:
    """Fires if EITHER a legal/explicit-escalation phrase is present, OR profanity/
    hostility appears near brand-directed language -- not on bare profanity alone."""
    pattern = "ABUSE_THREAT_ESCALATION_DEMAND (v1.1: legal/demand OR brand-directed hostility)"

    def search(self, text):
        return _ABUSE_LEGAL_OR_DEMAND.search(text) or _ABUSE_BRAND_DIRECTED_PROFANITY.search(text)


RULES = [
    ("ABUSE_THREAT_ESCALATION_DEMAND", _AbuseRule()),
    ("REFUND_RETURN", re.compile(
        r"\b(refund|return|replace(ment)?|money back|reimburse|compensat\w*)\b", re.I)),
    ("DELIVERY_QUALITY_ISSUE", re.compile(
        r"\b(damaged|broken|wrong item|wrong address|not\s*received.*delivered|"
        r"delivered.*(wrong|never|didn.?t (get|receive))|missing item)\b", re.I)),
    ("ORDER_STATUS", re.compile(
        r"\b(where is my order|track(ing)?|eta|hasn.?t (arrived|shipped)|order id|"
        r"delivery date|still (hasn|haven).?t|when will.*(arrive|deliver))\b", re.I)),
    ("ACCOUNT_LOGIN", re.compile(
        r"\b(log ?in|password|2.?step|verification code|account (blocked|locked|suspended)|"
        r"can.?t (sign|log) in)\b", re.I)),
    ("BILLING_PAYMENT", re.compile(
        r"\b(charged|charge\b|payment method|card declined|gift card|billed|double charged)\b", re.I)),
    ("PRIME_MEMBERSHIP", re.compile(r"\bprime\b", re.I)),
    ("DIGITAL_CONTENT_DEVICE", re.compile(
        r"\b(kindle|echo|alexa|fire ?tv|prime video|amazon music|audible|app (crash|won.?t))\b", re.I)),
    ("PRODUCT_INFO_QUESTION", re.compile(r"^(does|is|can|will|what|how)\b.*\?$", re.I)),
    ("ACKNOWLEDGEMENT_FOLLOWUP", re.compile(
        r"^(yes|no|ok(ay)?|thanks?|thank you|done|sent|details sent)\.?!?$", re.I)),
    ("COMPLAINT_SERVICE_QUALITY", re.compile(
        r"\b(worst|terrible|awful|pathetic|poor service|bad service|no help|unhelpful|"
        r"disappoint\w*|customer service)\b", re.I)),
]


def classify(text: str) -> dict:
    """Returns {intent, confidence, matched_rule}. Confidence is a fixed heuristic score
    per rule tier, NOT a calibrated probability -- see EVALUATION section in README for why
    we don't over-claim calibration for the rule baseline."""
    if not text or not text.strip():
        return {"intent": "OTHER_UNCLEAR", "confidence": 0.5, "matched_rule": None}
    for intent_id, pattern in RULES:
        if pattern.search(text):
            desc = pattern.pattern if hasattr(pattern, "pattern") else str(pattern)
            return {"intent": intent_id, "confidence": 0.75, "matched_rule": desc[:60]}
    return {"intent": "OTHER_UNCLEAR", "confidence": 0.4, "matched_rule": None}


if __name__ == "__main__":
    tests = [
        "Where is my order? It was supposed to arrive yesterday",
        "I want a refund for this damaged item",
        "Can't log in, not getting my 2 step verification code",
        "Thanks!",
        "This is fraud, I'm calling my lawyer",
    ]
    for t in tests:
        print(t, "->", classify(t))
