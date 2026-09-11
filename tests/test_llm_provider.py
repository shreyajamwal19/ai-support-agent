"""Tests for provider selection logic ONLY -- no live API calls (no network access to
LLM providers from the test/build environment, and tests must not depend on real keys).
"""
import os
import pytest
import sys
sys.path.insert(0, ".")
from src.llm.provider import get_active_provider_info, generate, LLMProviderError, MODELS


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ["LLM_PROVIDER", "LLM_MODEL", "GEMINI_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"]:
        monkeypatch.delenv(var, raising=False)
    yield


def test_defaults_to_gemini(monkeypatch):
    info = get_active_provider_info()
    assert info["provider"] == "gemini"
    assert info["model"] == MODELS["gemini"]
    assert info["api_key_present"] is False


def test_switch_to_openai_via_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    info = get_active_provider_info()
    assert info["provider"] == "openai"
    assert info["model"] == MODELS["openai"]


def test_switch_to_groq_via_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    info = get_active_provider_info()
    assert info["provider"] == "groq"


def test_invalid_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not_a_real_provider")
    with pytest.raises(LLMProviderError):
        get_active_provider_info()


def test_model_override_respected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-custom-model")
    info = get_active_provider_info()
    assert info["model"] == "gemini-custom-model"


def test_generate_fails_clearly_when_key_missing_no_silent_fallback(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    # no GEMINI_API_KEY set
    with pytest.raises(LLMProviderError, match="GEMINI_API_KEY"):
        generate("system", "user")


def test_generate_does_not_fall_back_to_another_provider_on_missing_key(monkeypatch):
    """Even with OTHER providers' keys present, an unset key for the SELECTED provider
    must fail -- never silently try a different provider."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test-only")
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test-only")
    with pytest.raises(LLMProviderError, match="GEMINI_API_KEY"):
        generate("system", "user")
