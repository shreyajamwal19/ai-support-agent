"""Phase 9: escalation policy. NOT a single confidence threshold -- an explicit,
inspectable set of risk signals, each producing its own named reason. Any signal firing
triggers escalation (OR logic): the philosophy stated in the assignment is that false
auto-handling can be much worse than an unnecessary escalation, so we bias toward
escalating on ambiguity in risk-relevant categories.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field

FINANCIAL_INTENTS = {"REFUND_RETURN", "BILLING_PAYMENT"}
SECURITY_INTENTS = {"ACCOUNT_LOGIN"}
HIGH_RISK_INTENTS = {"ABUSE_THREAT_ESCALATION_DEMAND"}

LEGAL_PATTERN = re.compile(
    r"\b(lawyer|attorney|sue|lawsuit|legal action|bbb|attorney general|regulator|fraud)\b", re.I)
IRREVERSIBLE_PATTERN = re.compile(
    r"\b(cancel my (account|order)|delete my account|close my account)\b", re.I)


@dataclass
class EscalationDecision:
    action: str  # "auto_handle" | "escalate"
    reason: str
    triggered_signals: list = field(default_factory=list)


def decide(intent: str, intent_confidence: float, retrieved_evidence: list,
           customer_text: str, evidence_conflict: bool = False) -> EscalationDecision:
    """Ordered, inspectable signal checks. First matching high-priority signal sets the
    primary `reason`, but all firing signals are recorded in `triggered_signals` for audit.
    """
    signals = []

    # 1. Legal/regulatory threat or explicit abuse -- always escalate regardless of intent
    if LEGAL_PATTERN.search(customer_text or ""):
        signals.append(("legal_regulatory_threat", "customer text references legal/regulatory action"))
    if intent in HIGH_RISK_INTENTS:
        signals.append(("abuse_or_explicit_escalation_demand", "intent classified as abuse/explicit escalation demand"))

    # 2. Insufficient grounding evidence -- can't confidently ground a reply
    if not retrieved_evidence or all(e["relevance"] < 0.3 for e in retrieved_evidence):
        signals.append(("insufficient_evidence", "no historical resolution retrieved above minimum relevance"))

    # 3. Conflicting historical evidence -- retrieved resolutions disagree materially
    if evidence_conflict:
        signals.append(("conflicting_evidence", "top retrieved historical resolutions conflict"))

    # 4. Low intent confidence
    if intent_confidence < 0.55:
        signals.append(("low_intent_confidence", f"intent confidence {intent_confidence:.2f} below 0.55 threshold"))

    # 5. Financial / refund implications -- irreversible money movement
    if intent in FINANCIAL_INTENTS:
        signals.append(("financial_refund_implication", "intent involves refund/billing action with real-money consequences"))

    # 6. Account security / identity verification
    if intent in SECURITY_INTENTS:
        signals.append(("account_security", "intent involves account access, requires identity verification"))

    # 7. Irreversible action requested
    if IRREVERSIBLE_PATTERN.search(customer_text or ""):
        signals.append(("irreversible_action_requested", "customer explicitly requests an irreversible account/order action"))

    if signals:
        primary = signals[0]
        return EscalationDecision(action="escalate", reason=primary[1],
                                    triggered_signals=[s[0] for s in signals])
    return EscalationDecision(action="auto_handle", reason="no risk signals triggered",
                                triggered_signals=[])


if __name__ == "__main__":
    print(decide("REFUND_RETURN", 0.8, [{"relevance": 0.6}], "I want a refund please"))
    print(decide("ORDER_STATUS", 0.9, [{"relevance": 0.7}], "where is my order"))
    print(decide("ABUSE_THREAT_ESCALATION_DEMAND", 0.9, [], "I'm calling my lawyer, this is fraud"))
    print(decide("ORDER_STATUS", 0.3, [], "hmm not sure whats going on"))
