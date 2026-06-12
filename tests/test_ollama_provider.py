"""Tests for the distinct, named Ollama provider adapter (2.1.0)."""

from __future__ import annotations

from unittest.mock import MagicMock

from omegaprompt.domain.enums import (
    OutputBudgetBucket,
    ReasoningProfile,
    ResponseSchemaMode,
)
from omegaprompt.providers import DEFAULT_MODELS, make_provider, supported_providers
from omegaprompt.providers.local_provider import LocalOpenAICompatibleProvider
from omegaprompt.providers.ollama_provider import OllamaProvider


def _request(**kw):
    from omegaprompt.providers import ProviderRequest

    base = {
        "system_prompt": "SP",
        "user_message": "UM",
        "reasoning_profile": ReasoningProfile.STANDARD,
        "output_budget_bucket": OutputBudgetBucket.MEDIUM,
        "response_schema_mode": ResponseSchemaMode.FREEFORM,
    }
    base.update(kw)
    return ProviderRequest(**base)


def test_ollama_in_supported_providers() -> None:
    assert "ollama" in supported_providers()


def test_make_provider_ollama_returns_named_class() -> None:
    p = make_provider("ollama", client=MagicMock())
    assert isinstance(p, OllamaProvider)
    assert p.name == "ollama"


def test_ollama_is_keyless_local_with_default_base_url() -> None:
    # No api_key, no base_url, no client -> constructs against the local
    # Ollama endpoint with a sentinel key (keyless from the user's view).
    p = OllamaProvider(model=DEFAULT_MODELS["ollama"], client=MagicMock())
    assert p.base_url == "http://localhost:11434/v1"
    assert p.model == DEFAULT_MODELS["ollama"]


def test_ollama_default_model() -> None:
    assert DEFAULT_MODELS["ollama"] == "llama3.1:8b"


def test_ollama_capabilities_are_local_experimental() -> None:
    p = make_provider("ollama", client=MagicMock())
    caps = p.capabilities()
    assert caps.provider == "ollama"
    assert caps.tier.value.startswith("tier_3")
    assert caps.experimental is True
    assert caps.ship_grade_judge is False
    assert caps.supports_llm_judge is False


def test_ollama_is_a_local_openai_compatible_subclass() -> None:
    # Backward-compatible: it still rides the OpenAI-compatible transport.
    assert issubclass(OllamaProvider, LocalOpenAICompatibleProvider)


def test_ollama_respects_explicit_base_url_and_key() -> None:
    p = OllamaProvider(
        model="llama3.1:8b",
        client=MagicMock(),
        api_key="custom",
        base_url="http://remote:11434/v1",
    )
    assert p.base_url == "http://remote:11434/v1"


def test_ollama_call_uses_openai_compatible_client() -> None:
    client = MagicMock()
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content="hello"), finish_reason="stop")]
    completion.usage = None
    client.chat.completions.create.return_value = completion
    p = make_provider("ollama", model="llama3.1:8b", client=client)
    resp = p.call(_request())
    assert resp.text == "hello"
    client.chat.completions.create.assert_called_once()
