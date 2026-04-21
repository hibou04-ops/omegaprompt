"""Provider factory + adapter tests. Mocked SDK clients, no network."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from omegaprompt.domain.enums import (
    OutputBudgetBucket,
    ReasoningProfile,
    ResponseSchemaMode,
)
from omegaprompt.domain.judge import Dimension, HardGate, JudgeResult, JudgeRubric
from omegaprompt.providers import (
    DEFAULT_MODELS,
    ProviderError,
    ProviderRequest,
    make_provider,
    supported_providers,
)
from omegaprompt.providers.anthropic_provider import AnthropicProvider
from omegaprompt.providers.base import empty_usage, normalize_usage
from omegaprompt.providers.openai_provider import OpenAIProvider


def _rubric() -> JudgeRubric:
    return JudgeRubric(
        dimensions=[Dimension(name="q", description="q", weight=1.0)],
        hard_gates=[HardGate(name="g", description="g")],
    )


def _request(**kw) -> ProviderRequest:
    base = {
        "system_prompt": "SP",
        "user_message": "UM",
        "few_shots": [],
        "reasoning_profile": ReasoningProfile.STANDARD,
        "output_budget_bucket": OutputBudgetBucket.MEDIUM,
        "response_schema_mode": ResponseSchemaMode.FREEFORM,
    }
    base.update(kw)
    return ProviderRequest(**base)


# --------------------------- factory -----------------------------


def test_supported_providers_lists_both():
    names = supported_providers()
    assert "anthropic" in names
    assert "openai" in names
    assert "gemini" in names
    assert "ollama" in names


def test_default_models_per_provider():
    assert DEFAULT_MODELS["anthropic"].startswith("claude-")
    assert DEFAULT_MODELS["openai"].startswith("gpt-")


def test_make_provider_unknown_raises():
    with pytest.raises(ProviderError, match="Unknown provider"):
        make_provider("bogus-provider")


def test_make_provider_uses_default_model():
    p = make_provider("anthropic", client=MagicMock())
    assert p.model == DEFAULT_MODELS["anthropic"]


def test_make_provider_openai_passes_base_url():
    p = make_provider("openai", client=MagicMock(), base_url="http://localhost:11434/v1")
    assert p.base_url == "http://localhost:11434/v1"


def test_make_provider_local_alias_has_local_capabilities():
    p = make_provider("ollama", client=MagicMock(), base_url="http://localhost:11434/v1")
    caps = p.capabilities()
    assert caps.tier.value.startswith("tier_3")
    assert caps.supports_llm_judge is False
    assert caps.experimental is True


# --------------------------- AnthropicProvider ---------------------------


def _anthropic_target_response(text: str = "hello"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


def _anthropic_structured_response(parsed: JudgeResult):
    return SimpleNamespace(
        parsed_output=parsed,
        stop_reason="end_turn",
        content=[],
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=4096,
        ),
    )


def test_anthropic_freeform_applies_thinking_for_deep():
    client = MagicMock()
    client.messages.create.return_value = _anthropic_target_response("ok")
    p = AnthropicProvider(model="claude-test", client=client)

    resp = p.call(_request(
        few_shots=[{"input": "a", "output": "b"}],
        reasoning_profile=ReasoningProfile.DEEP,
        output_budget_bucket=OutputBudgetBucket.LARGE,
    ))
    assert resp.text == "ok"
    kw = client.messages.create.call_args.kwargs
    assert kw["model"] == "claude-test"
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"] == {"effort": "high"}
    assert kw["max_tokens"] == 16000
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kw["messages"][0] == {"role": "user", "content": "a"}
    assert kw["messages"][1] == {"role": "assistant", "content": "b"}
    assert kw["messages"][2] == {"role": "user", "content": "UM"}


def test_anthropic_freeform_drops_thinking_when_off():
    client = MagicMock()
    client.messages.create.return_value = _anthropic_target_response()
    p = AnthropicProvider(model="x", client=client)
    p.call(_request(reasoning_profile=ReasoningProfile.OFF, output_budget_bucket=OutputBudgetBucket.SMALL))
    kw = client.messages.create.call_args.kwargs
    assert "thinking" not in kw
    assert "output_config" not in kw
    assert kw["max_tokens"] == 1024


def test_anthropic_strict_schema_uses_parse():
    client = MagicMock()
    expected = JudgeResult(scores={"q": 4}, gate_results={"g": True})
    client.messages.parse.return_value = _anthropic_structured_response(expected)
    p = AnthropicProvider(model="claude-j", client=client)

    resp = p.call(_request(
        response_schema_mode=ResponseSchemaMode.STRICT_SCHEMA,
        output_schema=JudgeResult,
    ))
    assert resp.parsed == expected
    assert resp.usage["cache_read_input_tokens"] == 4096
    kw = client.messages.parse.call_args.kwargs
    assert kw["output_format"] is JudgeResult
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_strict_schema_without_output_schema_raises():
    p = AnthropicProvider(model="x", client=MagicMock())
    with pytest.raises(ProviderError, match="requires request.output_schema"):
        p.call(_request(response_schema_mode=ResponseSchemaMode.STRICT_SCHEMA))


def test_anthropic_strict_raises_on_refusal():
    client = MagicMock()
    client.messages.parse.return_value = SimpleNamespace(
        parsed_output=None, stop_reason="refusal", content=[], usage=None
    )
    p = AnthropicProvider(model="x", client=client)
    with pytest.raises(ProviderError, match="refused"):
        p.call(_request(
            response_schema_mode=ResponseSchemaMode.STRICT_SCHEMA,
            output_schema=JudgeResult,
        ))


def test_anthropic_json_object_adds_suffix_to_system():
    client = MagicMock()
    client.messages.create.return_value = _anthropic_target_response()
    p = AnthropicProvider(model="x", client=client)
    p.call(_request(response_schema_mode=ResponseSchemaMode.JSON_OBJECT))
    kw = client.messages.create.call_args.kwargs
    assert "valid JSON object" in kw["system"][0]["text"]


# ---------------------------- OpenAIProvider -----------------------------


def _openai_target_response(text: str = "ok"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=20,
            completion_tokens=10,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
    )


def _openai_structured_response(parsed, finish_reason="stop"):
    message = SimpleNamespace(parsed=parsed)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=SimpleNamespace(
            prompt_tokens=200,
            completion_tokens=80,
            prompt_tokens_details=SimpleNamespace(cached_tokens=4096),
        ),
    )


def test_openai_freeform_builds_messages():
    client = MagicMock()
    client.chat.completions.create.return_value = _openai_target_response("reply")
    p = OpenAIProvider(model="gpt-test", client=client)
    resp = p.call(_request(
        few_shots=[{"input": "a", "output": "b"}],
        reasoning_profile=ReasoningProfile.OFF,
    ))
    assert resp.text == "reply"
    assert resp.usage["input_tokens"] == 20
    kw = client.chat.completions.create.call_args.kwargs
    assert kw["model"] == "gpt-test"
    assert kw["messages"][0] == {"role": "system", "content": "SP"}
    assert kw["messages"][1] == {"role": "user", "content": "a"}
    assert kw["messages"][2] == {"role": "assistant", "content": "b"}
    assert kw["messages"][3] == {"role": "user", "content": "UM"}
    assert "reasoning_effort" not in kw


def test_openai_freeform_adds_reasoning_effort_when_enabled():
    client = MagicMock()
    client.chat.completions.create.return_value = _openai_target_response()
    p = OpenAIProvider(model="gpt-x", client=client)
    p.call(_request(reasoning_profile=ReasoningProfile.DEEP))
    kw = client.chat.completions.create.call_args.kwargs
    assert kw["reasoning_effort"] == "high"


def test_openai_json_object_sets_response_format():
    client = MagicMock()
    client.chat.completions.create.return_value = _openai_target_response()
    p = OpenAIProvider(model="gpt-x", client=client)
    p.call(_request(response_schema_mode=ResponseSchemaMode.JSON_OBJECT))
    kw = client.chat.completions.create.call_args.kwargs
    assert kw["response_format"] == {"type": "json_object"}


def test_openai_strict_schema_uses_beta_parse():
    client = MagicMock()
    expected = JudgeResult(scores={"q": 3}, gate_results={"g": True})
    client.beta.chat.completions.parse.return_value = _openai_structured_response(expected)
    p = OpenAIProvider(model="gpt-j", client=client)
    resp = p.call(_request(
        response_schema_mode=ResponseSchemaMode.STRICT_SCHEMA,
        output_schema=JudgeResult,
    ))
    assert resp.parsed == expected
    assert resp.usage["cache_read_input_tokens"] == 4096
    assert resp.usage["input_tokens"] == 200
    kw = client.beta.chat.completions.parse.call_args.kwargs
    assert kw["response_format"] is JudgeResult


def test_openai_strict_raises_on_content_filter():
    client = MagicMock()
    client.beta.chat.completions.parse.return_value = _openai_structured_response(
        JudgeResult(scores={"q": 1}, gate_results={"g": True}),
        finish_reason="content_filter",
    )
    p = OpenAIProvider(model="x", client=client)
    with pytest.raises(ProviderError, match="content_filter"):
        p.call(_request(
            response_schema_mode=ResponseSchemaMode.STRICT_SCHEMA,
            output_schema=JudgeResult,
        ))


def test_openai_strict_raises_when_parsed_missing():
    client = MagicMock()
    client.beta.chat.completions.parse.return_value = _openai_structured_response(None)
    p = OpenAIProvider(model="x", client=client)
    with pytest.raises(ProviderError, match="no parsed object"):
        p.call(_request(
            response_schema_mode=ResponseSchemaMode.STRICT_SCHEMA,
            output_schema=JudgeResult,
        ))


def test_local_provider_expedition_falls_back_from_strict_schema():
    client = MagicMock()
    client.chat.completions.create.return_value = _openai_target_response('{"scores":{"q":3},"gate_results":{"g":true},"notes":"ok"}')
    p = make_provider(
        "ollama",
        model="llama3.1:8b",
        client=client,
        base_url="http://localhost:11434/v1",
    )
    resp = p.call(_request(
        response_schema_mode=ResponseSchemaMode.STRICT_SCHEMA,
        output_schema=JudgeResult,
        execution_profile="expedition",
    ))
    assert isinstance(resp.parsed, JudgeResult)
    assert any(event.capability == "structured_output" for event in resp.degraded_capabilities)


# --------------------------- normalize_usage -----------------------------


def test_normalize_usage_anthropic_shape():
    raw = SimpleNamespace(
        input_tokens=10,
        output_tokens=5,
        cache_creation_input_tokens=100,
        cache_read_input_tokens=200,
    )
    u = normalize_usage(raw)
    assert u == {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_creation_input_tokens": 100,
        "cache_read_input_tokens": 200,
    }


def test_normalize_usage_openai_shape():
    raw = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        prompt_tokens_details=SimpleNamespace(cached_tokens=7),
    )
    u = normalize_usage(raw)
    assert u["input_tokens"] == 10
    assert u["output_tokens"] == 5
    assert u["cache_read_input_tokens"] == 7


def test_normalize_usage_none():
    assert normalize_usage(None) == empty_usage()
