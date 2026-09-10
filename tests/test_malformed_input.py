import sys
sys.path.insert(0, ".")
from src.intent.rule_classifier import classify
from src.escalation.policy import decide
from src.generation.responder import draft_extractive, grounding_check


def test_pipeline_components_handle_empty_string():
    r = classify("")
    d = decide(r["intent"], r["confidence"], [], "")
    g = draft_extractive("", [], r["intent"])
    assert d.action in {"auto_handle", "escalate"}
    assert isinstance(g["draft_reply"], str)


def test_pipeline_components_handle_very_long_text():
    long_text = "help " * 5000
    r = classify(long_text)
    assert r["intent"] is not None


def test_grounding_check_handles_empty_evidence():
    result = grounding_check("we will refund you", [])
    assert result["passed"] is False
