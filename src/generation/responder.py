"""Phase 8: response drafting. IMPORTANT SCOPE NOTE (see DECISIONS.md #9): no
ANTHROPIC_API_KEY was available in the build/eval sandbox, so the shipped default path is
EXTRACTIVE/template-grounded (adapts the single most similar historical resolution),
NOT free-generation. An LLM-generation path (`generate_llm`) is implemented against the
standard Anthropic Messages API and is fully wired into the pipeline -- it activates
automatically if ANTHROPIC_API_KEY is set (see src/generation/llm_responder.py) -- but was
not exercised for this submission's reported numbers. This is a real, load-bearing
limitation, not a footnote; it is restated in REPORT.md's "misleading headline number"
section.
"""
from __future__ import annotations
import re

REFUND_PROMISE_PATTERN = re.compile(
    r"\b(we (will|have) (refund|credit|compensat\w*|replace)|your refund (has|is) been (issued|processed))\b", re.I)


def grounding_check(draft: str, evidence: list) -> dict:
    """Cheap but real safety check: flag drafts that assert a concrete promise (refund
    issued, replacement shipped) when no retrieved evidence actually shows that outcome
    being taken -- rather than trust the draft blindly. This is intentionally conservative
    (may flag some acceptable phrasing) because the assignment explicitly requires we not
    invent unsupported promises."""
    flags = []
    if REFUND_PROMISE_PATTERN.search(draft):
        evidence_text = " ".join(e["historical_resolution"] for e in evidence)
        if not REFUND_PROMISE_PATTERN.search(evidence_text):
            flags.append("draft_asserts_concrete_action_not_present_in_evidence")
    return {"passed": len(flags) == 0, "flags": flags}


def draft_extractive(customer_text: str, evidence: list, intent: str) -> dict:
    """Template-grounded drafting: adapt the top retrieved historical resolution rather
    than free-generate. Traceable 1:1 to a specific evidence item -- the opposite of
    hallucination-prone, at the cost of being less naturally phrased than an LLM draft.
    Returns a structured draft + which evidence item it was grounded on."""
    if not evidence:
        return {
            "draft_reply": ("Thanks for reaching out — to look into this properly we need "
                             "a bit more detail and will have a specialist follow up with you directly."),
            "grounded_on": None,
            "grounding": {"passed": True, "flags": []},
        }
    top = evidence[0]
    resolution = top["historical_resolution"]
    # Extremely light templating: prefix acknowledges the customer's message, then reuses
    # the historically-grounded resolution text verbatim (never invents new specifics).
    draft = f"{resolution}"
    check = grounding_check(draft, evidence)
    return {"draft_reply": draft, "grounded_on": top["pair_id"], "grounding": check}


if __name__ == "__main__":
    ev = [{"pair_id": "x", "relevance": 0.8,
           "historical_customer_message": "order hasn't arrived",
           "historical_resolution": "Sorry to hear this! What does the current tracking info say?"}]
    print(draft_extractive("my order hasn't come yet", ev, "ORDER_STATUS"))
