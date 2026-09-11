"""LLM-based drafting path, provider-agnostic (see src/llm/provider.py: Gemini primary,
OpenAI/Groq switchable via LLM_PROVIDER env var). NOT executed for this submission's
reported metrics -- no provider API key is available in the build sandbox and none of
Gemini/OpenAI/Groq's endpoints are reachable from it either. See DECISIONS.md.
"""
import json
import sys
sys.path.insert(0, ".")
from src.llm.provider import generate

SYSTEM_PROMPT = """You are drafting a customer support reply for AmazonHelp on Twitter.
Rules:
- Only state facts/actions supported by the provided historical resolution evidence.
- Never promise a specific refund/credit/replacement unless the evidence shows that action
  was actually taken in a similar case.
- If evidence is weak or conflicting, ask a clarifying question instead of guessing.
- Keep the reply under 280 characters, in the brand's typical tone (empathetic, concise).
Return ONLY JSON: {"draft_reply": str, "used_evidence_ids": [str], "confidence": float}
"""

def generate_reply(customer_text: str, evidence: list, intent: str) -> dict:
    """Returns the parsed draft dict PLUS provider_info so callers can stamp results with
    exactly which provider/model produced it -- required per DECISIONS.md provider policy."""
    user_msg = json.dumps({"customer_message": customer_text, "intent": intent,
                            "historical_evidence": evidence})
    text, provider_info = generate(SYSTEM_PROMPT, user_msg, max_tokens=300)
    parsed = json.loads(text)
    parsed["provider_info"] = provider_info
    return parsed
