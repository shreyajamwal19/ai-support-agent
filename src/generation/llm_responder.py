"""LLM-based drafting path using the standard Anthropic Messages API. Requires
ANTHROPIC_API_KEY in the environment. NOT executed for this submission's reported metrics
(no key was available in the build sandbox) -- see DECISIONS.md #9 and REPORT.md.
"""
import os
import json

SYSTEM_PROMPT = """You are drafting a customer support reply for AmazonHelp on Twitter.
Rules:
- Only state facts/actions supported by the provided historical resolution evidence.
- Never promise a specific refund/credit/replacement unless the evidence shows that action
  was actually taken in a similar case.
- If evidence is weak or conflicting, ask a clarifying question instead of guessing.
- Keep the reply under 280 characters, in the brand's typical tone (empathetic, concise).
Return ONLY JSON: {"draft_reply": str, "used_evidence_ids": [str], "confidence": float}
"""

def generate(customer_text: str, evidence: list, intent: str) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set -- LLM generation path unavailable in this "
            "environment. Falling back path is src.generation.responder.draft_extractive()."
        )
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    user_msg = json.dumps({"customer_message": customer_text, "intent": intent,
                            "historical_evidence": evidence})
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return json.loads(text)
