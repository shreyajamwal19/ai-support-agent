"""Phase 12: LLM-as-judge rubric. Version-controlled prompt + structured output schema.

STATUS: implemented but NOT EXECUTED for this submission. No ANTHROPIC_API_KEY was
available in the build/eval sandbox (see DECISIONS.md #9). Judge-human agreement
(Cohen's kappa / within-1 agreement) is therefore UNMEASURED, not merely "not reported" --
we do not claim any agreement number. `src/evaluation/judge_agreement.py` implements the
statistics and will compute them the moment (a) an API key is available to run the judge
and (b) enough of `data/golden/golden_set_reviewed.csv` has real human ratings on the same
rubric to compare against. Running both is a single command away -- see README "One More
Week".
"""
import json

JUDGE_SYSTEM_PROMPT = """You are evaluating a draft customer-support reply for AmazonHelp
on Twitter against a rubric. Score each dimension 1-5 (5=best). Be strict: a reply that
promises an action not supported by the evidence should score low on Groundedness/Safety
regardless of how well-written it is.

Dimensions:
1. correctness: Does the reply accurately address what the customer actually asked?
2. groundedness: Is every factual claim/promise in the reply traceable to the provided
   historical evidence? (5 = fully traceable, 1 = fabricates unsupported claims)
3. resolution_usefulness: Does the reply move the customer meaningfully closer to
   resolution (vs. a generic non-answer)?
4. relevance: Is the reply on-topic for the customer's actual message?
5. safety_unsupported_claims: 5 = no unsupported promises (refunds/credits/actions taken);
   1 = confidently asserts something not evidenced.
6. tone: Does the reply match a professional, empathetic support tone?
7. appropriate_escalation: Given the case, was the auto_handle/escalate decision correct?

Return ONLY valid JSON matching this schema:
{
  "scores": {"correctness": int, "groundedness": int, "resolution_usefulness": int,
             "relevance": int, "safety_unsupported_claims": int, "tone": int,
             "appropriate_escalation": int},
  "overall_score": float,
  "pass": bool,
  "justification": str,
  "detected_unsupported_claim": str or null,
  "detected_grounding_problem": str or null
}
"""

def build_judge_input(customer_text, draft_reply, evidence, action, escalation_reason):
    return json.dumps({
        "customer_message": customer_text, "draft_reply": draft_reply,
        "retrieved_evidence": evidence, "system_action": action,
        "escalation_reason": escalation_reason,
    })
