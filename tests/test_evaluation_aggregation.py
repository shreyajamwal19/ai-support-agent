import sys
sys.path.insert(0, ".")
from src.evaluation.run import eval_intent_classifier, eval_escalation


def test_intent_eval_perfect_predictions():
    gold = ["A", "B", "A", "C"]
    preds = ["A", "B", "A", "C"]
    r = eval_intent_classifier("perfect", preds, gold)
    assert r["accuracy"] == 1.0
    assert r["macro_f1"] == 1.0


def test_intent_eval_all_wrong():
    gold = ["A", "B"]
    preds = ["B", "A"]
    r = eval_intent_classifier("wrong", preds, gold)
    assert r["accuracy"] == 0.0


def test_escalation_eval_harmful_auto_handle_detected():
    gold = ["escalate", "auto_handle"]
    preds = ["auto_handle", "auto_handle"]
    r = eval_escalation(preds, gold)
    assert r["harmful_auto_handle_count"] == 1
    assert r["harmful_auto_handle_rate"] == 0.5


def test_escalation_eval_unnecessary_escalation_detected():
    gold = ["auto_handle", "auto_handle"]
    preds = ["escalate", "auto_handle"]
    r = eval_escalation(preds, gold)
    assert r["unnecessary_escalation_count"] == 1
