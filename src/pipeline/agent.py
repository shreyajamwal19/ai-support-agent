"""Phase 5: end-to-end pipeline wiring intent -> retrieval -> generation -> escalation ->
structured output. This is the object the evaluation harness and README quickstart call.
"""
from __future__ import annotations
import sys
sys.path.insert(0, ".")
from src.intent.rule_classifier import classify as rule_classify
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.generation.responder import draft_extractive
from src.escalation.policy import decide as escalation_decide


class SupportAgent:
    def __init__(self, retriever: TfidfRetriever | None = None):
        self.retriever = retriever or TfidfRetriever().load()

    def handle(self, customer_text: str, exclude_pair_ids=None) -> dict:
        intent_result = rule_classify(customer_text)
        evidence = self.retriever.query(customer_text, k=3,
                                          exclude_pair_ids=exclude_pair_ids)
        gen = draft_extractive(customer_text, evidence, intent_result["intent"])
        esc = escalation_decide(
            intent=intent_result["intent"],
            intent_confidence=intent_result["confidence"],
            retrieved_evidence=evidence,
            customer_text=customer_text,
        )
        # generation-grounding failure forces escalation even if the policy above didn't
        # already catch it -- never ship an unsupported-claim draft as auto_handle.
        action = esc.action
        reason = esc.reason
        if not gen["grounding"]["passed"] and action == "auto_handle":
            action = "escalate"
            reason = "draft_failed_grounding_check: " + ",".join(gen["grounding"]["flags"])

        return {
            "intent": intent_result["intent"],
            "intent_confidence": intent_result["confidence"],
            "action": action,
            "escalation_reason": reason if action == "escalate" else None,
            "draft_reply": gen["draft_reply"],
            "evidence": evidence,
            "response_confidence": round(evidence[0]["relevance"], 3) if evidence else 0.0,
        }


if __name__ == "__main__":
    agent = SupportAgent()
    for text in ["where is my order it still hasn't arrived",
                 "this is fraud I'm calling my lawyer, refund me now",
                 "asdkjaslkdj random text"]:
        import json
        print(json.dumps(agent.handle(text), indent=2))
