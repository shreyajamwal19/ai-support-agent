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


def test_known_high_confidence_semantic_mismatch_case_reproduces():
    """Regression/documentation test for REPORT.md Failure Analysis #5: a real case where
    TF-IDF gives high lexical relevance (>0.7) to a semantically wrong historical match
    ('pick it up' pre-purchase question vs. 'pickup arrangement' return complaint), and the
    grounding check does NOT catch it (it only detects unsupported promise-language, not
    topical mismatch). This test locks in that the gap is real, reproducible, and
    currently unmitigated -- not an anecdote -- so a future fix has something concrete to
    verify against.
    """
    import sys
    sys.path.insert(0, ".")
    from pathlib import Path
    if not Path("artifacts/index/tfidf_retriever.pkl").exists():
        import pytest
        pytest.skip("fitted retriever index not present in this environment")
    from src.pipeline.agent import SupportAgent
    agent = SupportAgent()
    out = agent.handle("Can I purchase something and have someone else pick it up?")
    assert out["evidence"], "expected at least one retrieval hit for this query"
    top = out["evidence"][0]
    # Documents the specific known gap: high relevance score alone is not sufficient
    # evidence of topical correctness, and the system currently ships this as auto_handle.
    assert top["relevance"] > 0.6, (
        "if this no longer reproduces at high relevance, the retrieval pool or "
        "vectorizer changed -- re-verify Failure Analysis #5 in REPORT.md against a "
        "fresh example rather than assuming the gap is fixed"
    )
    assert out["action"] == "auto_handle", (
        "documents that this specific known-bad case currently ships without escalation; "
        "if this now escalates, REPORT.md Failure Analysis #5's 'unmitigated' claim is "
        "stale and should be updated, not silently left inconsistent"
    )
