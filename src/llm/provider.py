"""Single-file, provider-agnostic LLM client. Deliberately minimal (per instruction: do
not over-engineer). Supports Gemini (primary), OpenAI, and Groq, selected by ONE
environment variable so a benchmark run always uses exactly one provider.

Environment variables:
  LLM_PROVIDER      "gemini" (default) | "openai" | "groq"
  GEMINI_API_KEY    required if LLM_PROVIDER=gemini
  OPENAI_API_KEY    required if LLM_PROVIDER=openai
  GROQ_API_KEY      required if LLM_PROVIDER=groq
  LLM_MODEL         optional override; each provider has a sane default (see MODELS below)

Design rules (see DECISIONS.md "provider selection"):
  - The provider is resolved ONCE per process, at import/first-use time, from env vars.
  - There is NO automatic fallback between providers. If the configured provider's API
    call fails, this raises -- it does not retry against a different provider. Mixing
    providers mid-benchmark would silently invalidate any judge-agreement or generation-
    quality comparison, so failure must be loud, not swallowed.
  - Every call returns (text, provider_info) where provider_info = {"provider": ...,
    "model": ...} so callers (the eval harness) can stamp results with exactly what
    generated them.
"""
from __future__ import annotations
import os


MODELS = {
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
}


class LLMProviderError(RuntimeError):
    """Raised on missing config or a failed call. Never caught-and-retried against a
    different provider automatically -- see module docstring."""


def _resolve_provider() -> str:
    provider = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
    if provider not in MODELS:
        raise LLMProviderError(
            f"LLM_PROVIDER={provider!r} is not supported. Choose one of: {list(MODELS)}"
        )
    return provider


def get_active_provider_info() -> dict:
    """What provider/model would be used right now, without making a call. Use this to
    stamp evaluation metadata BEFORE a benchmark run starts, so provider is recorded even
    if the run later fails partway through."""
    provider = _resolve_provider()
    model = os.environ.get("LLM_MODEL", MODELS[provider])
    key_env = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY"}[provider]
    return {
        "provider": provider,
        "model": model,
        "api_key_present": bool(os.environ.get(key_env)),
    }


def generate(system_prompt: str, user_prompt: str, max_tokens: int = 400) -> tuple[str, dict]:
    """Single entry point every caller (generation, judge) should use. Resolves the
    provider fresh on every call from env vars (so a long-running process always reflects
    the current env), but does NOT fall back to a different provider if the call fails --
    it raises LLMProviderError instead.
    """
    info = get_active_provider_info()
    provider, model = info["provider"], info["model"]
    if not info["api_key_present"]:
        key_env = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY"}[provider]
        raise LLMProviderError(
            f"LLM_PROVIDER={provider} but {key_env} is not set. Set it in your local "
            f".env (never commit it) or switch LLM_PROVIDER to a provider you have a key for."
        )

    try:
        if provider == "gemini":
            text = _call_gemini(system_prompt, user_prompt, model, max_tokens)
        elif provider == "openai":
            text = _call_openai(system_prompt, user_prompt, model, max_tokens)
        elif provider == "groq":
            text = _call_groq(system_prompt, user_prompt, model, max_tokens)
        else:  # unreachable given _resolve_provider's check, kept for clarity
            raise LLMProviderError(f"unhandled provider {provider}")
    except LLMProviderError:
        raise
    except Exception as e:
        # Fail clearly and loudly -- do NOT catch this and try another provider.
        raise LLMProviderError(f"{provider} call failed: {e}") from e

    return text, {"provider": provider, "model": model}


def _call_gemini(system_prompt: str, user_prompt: str, model: str, max_tokens: int) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    client = genai.GenerativeModel(model, system_instruction=system_prompt)
    resp = client.generate_content(
        user_prompt,
        generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens),
    )
    return resp.text


def _call_openai(system_prompt: str, user_prompt: str, model: str, max_tokens: int) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system_prompt},
                   {"role": "user", "content": user_prompt}],
    )
    return resp.choices[0].message.content


def _call_groq(system_prompt: str, user_prompt: str, model: str, max_tokens: int) -> str:
    # Groq's API is OpenAI-compatible; reuse the openai SDK pointed at Groq's base URL.
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system_prompt},
                   {"role": "user", "content": user_prompt}],
    )
    return resp.choices[0].message.content
