import sys
sys.path.insert(0, ".")
from src.intent.rule_classifier import classify, INTENT_IDS


def test_empty_text_returns_other_unclear():
    r = classify("")
    assert r["intent"] == "OTHER_UNCLEAR"


def test_none_text_handled_gracefully():
    r = classify(None)
    assert r["intent"] == "OTHER_UNCLEAR"


def test_refund_keyword_detected():
    r = classify("I want a refund for this")
    assert r["intent"] == "REFUND_RETURN"


def test_all_taxonomy_ids_are_reachable_in_rules():
    import src.intent.rule_classifier as rc
    rule_intents = {intent for intent, _ in rc.RULES}
    rule_intents.add("OTHER_UNCLEAR")  # fallback
    missing = set(INTENT_IDS) - rule_intents
    assert missing == set(), f"Taxonomy intents with no rule coverage: {missing}"


def test_malformed_unicode_does_not_crash():
    classify("\ud83d\ude00 test emoji garbage \x00\x01")


def test_v1_1_bare_profanity_not_directed_at_brand_is_not_abuse():
    """Failure Analysis #4 fix: 'the piece of shit doesn't work' is a product complaint,
    not brand-directed abuse -- should classify by topic (delivery/quality), not abuse."""
    r = classify("the piece of shit doesnt even work")
    assert r["intent"] != "ABUSE_THREAT_ESCALATION_DEMAND"


def test_v1_1_brand_directed_profanity_is_still_abuse():
    r = classify("your customer service is fucking useless")
    assert r["intent"] == "ABUSE_THREAT_ESCALATION_DEMAND"


def test_v1_1_legal_threat_still_detected_regardless_of_proximity():
    r = classify("I am going to sue you, this is fraud")
    assert r["intent"] == "ABUSE_THREAT_ESCALATION_DEMAND"
