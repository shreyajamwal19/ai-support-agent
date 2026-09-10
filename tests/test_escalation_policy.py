import sys
sys.path.insert(0, ".")
from src.escalation.policy import decide


def test_legal_threat_always_escalates():
    d = decide("ORDER_STATUS", 0.95, [{"relevance": 0.9}], "I'm calling my lawyer, this is fraud")
    assert d.action == "escalate"
    assert "legal" in d.reason


def test_high_confidence_grounded_order_status_auto_handles():
    d = decide("ORDER_STATUS", 0.9, [{"relevance": 0.8}], "where is my order")
    assert d.action == "auto_handle"


def test_no_evidence_escalates():
    d = decide("ORDER_STATUS", 0.9, [], "where is my order")
    assert d.action == "escalate"
    assert "insufficient_evidence" in d.triggered_signals


def test_financial_intent_always_escalates_even_with_high_confidence():
    d = decide("REFUND_RETURN", 0.95, [{"relevance": 0.9}], "please refund me")
    assert d.action == "escalate"
    assert "financial_refund_implication" in d.triggered_signals


def test_account_security_escalates():
    d = decide("ACCOUNT_LOGIN", 0.9, [{"relevance": 0.9}], "cant log in")
    assert d.action == "escalate"
    assert "account_security" in d.triggered_signals


def test_low_confidence_escalates():
    d = decide("OTHER_UNCLEAR", 0.2, [{"relevance": 0.5}], "hmm")
    assert "low_intent_confidence" in d.triggered_signals
