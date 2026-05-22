"""Provider compatibility contracts with no live provider calls.

These tests are intentionally broader than adapter unit tests. They lock the
public capability claims, guarded/expedition fallback boundaries, and artifact
degradation propagation that README/docs are allowed to describe.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from omegaprompt.dataset import Dataset
from omegaprompt.domain.enums import ReasoningProfile, ResponseSchemaMode
from omegaprompt.domain.judge import Dimension, HardGate, JudgeResult, JudgeRubric
from omegaprompt.domain.params import PromptVariants
from omegaprompt.domain.profiles import ExecutionProfile
from omegaprompt.domain.result import CalibrationArtifact
from omegaprompt.judges.base import JudgeOutcome
from omegaprompt.providers import DEFAULT_MODELS, make_provider, supported_providers
from omegaprompt.providers.anthropic_provider import AnthropicProvider
from omegaprompt.providers.base import (
    CapabilityEvent,
    CapabilityTier,
    ProviderCapabilities,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    normalize_usage,
    provider_capabilities,
)
from omegaprompt.providers.gemini_provider import GeminiProvider
from omegaprompt.providers.openai_provider import OpenAIProvider
from omegaprompt.targets.prompt_target import PromptTarget


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PROVIDERS = {
    "anthropic",
    "openai",
    "gemini",
    "local",
    "ollama",
    "vllm",
    "llama_cpp",
}


def _request(**overrides: object) -> ProviderRequest:
    data = {
        "system_prompt": "SP",
        "user_message": "UM",
        "reasoning_profile": ReasoningProfile.OFF,
        "response_schema_mode": ResponseSchemaMode.FREEFORM,
        "execution_profile": ExecutionProfile.GUARDED,
    }
    data.update(overrides)
    return ProviderRequest(**data)


def _openai_target_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=13, completion_tokens=5),
    )


def _rubric() -> JudgeRubric:
    return JudgeRubric(
        dimensions=[Dimension(name="quality", description="Quality.", weight=1.0)],
        hard_gates=[HardGate(name="correctness", description="Correct.", evaluator="judge")],
    )


def test_provider_factory_registry_and_default_models_are_explicit() -> None:
    assert set(supported_providers()) == CONTRACT_PROVIDERS
    assert set(DEFAULT_MODELS) == CONTRACT_PROVIDERS
    assert all(DEFAULT_MODELS[name] for name in CONTRACT_PROVIDERS)

    for provider_name in CONTRACT_PROVIDERS:
        provider = make_provider(
            provider_name,
            client=MagicMock(),
            base_url="http://localhost:11434/v1",
        )
        assert provider.name == provider_name
        assert provider.model == DEFAULT_MODELS[provider_name]


@pytest.mark.parametrize(
    ("provider_name", "expected"),
    [
        (
            "anthropic",
            {
                "tier": CapabilityTier.CLOUD,
                "strict": True,
                "judge": True,
                "ship": True,
                "experimental": False,
                "placeholder": False,
            },
        ),
        (
            "openai",
            {
                "tier": CapabilityTier.CLOUD,
                "strict": True,
                "judge": True,
                "ship": True,
                "experimental": False,
                "placeholder": False,
            },
        ),
        (
            "gemini",
            {
                "tier": CapabilityTier.CLOUD,
                "strict": True,
                "judge": True,
                "ship": False,
                "experimental": False,
                "placeholder": False,
            },
        ),
        (
            "local",
            {
                "tier": CapabilityTier.LOCAL,
                "strict": False,
                "judge": False,
                "ship": False,
                "experimental": True,
                "placeholder": False,
            },
        ),
        (
            "ollama",
            {
                "tier": CapabilityTier.LOCAL,
                "strict": False,
                "judge": False,
                "ship": False,
                "experimental": True,
                "placeholder": False,
            },
        ),
        (
            "vllm",
            {
                "tier": CapabilityTier.LOCAL,
                "strict": False,
                "judge": False,
                "ship": False,
                "experimental": True,
                "placeholder": False,
            },
        ),
        (
            "llama_cpp",
            {
                "tier": CapabilityTier.LOCAL,
                "strict": False,
                "judge": False,
                "ship": False,
                "experimental": True,
                "placeholder": False,
            },
        ),
    ],
)
def test_provider_capabilities_contract_by_provider(
    provider_name: str,
    expected: dict[str, object],
) -> None:
    kwargs: dict[str, object] = {"client": MagicMock()}
    if provider_name in {"local", "ollama", "vllm", "llama_cpp"}:
        kwargs["base_url"] = "http://localhost:11434/v1"
    provider = make_provider(provider_name, **kwargs)
    caps = provider_capabilities(provider)

    assert caps.provider == provider_name
    assert caps.tier == expected["tier"]
    assert caps.supports_strict_schema is expected["strict"]
    assert caps.supports_json_object is True
    assert caps.supports_llm_judge is expected["judge"]
    assert caps.ship_grade_judge is expected["ship"]
    assert caps.experimental is expected["experimental"]
    assert caps.placeholder is expected["placeholder"]


def test_legacy_provider_capabilities_fail_closed_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OMEGAPROMPT_TRUST_LEGACY_PROVIDERS", raising=False)

    class LegacyProvider:
        name = "legacy-contract-provider"

    caps = provider_capabilities(LegacyProvider())

    assert caps.provider == "legacy-contract-provider"
    assert caps.experimental is True
    assert caps.supports_strict_schema is False
    assert caps.supports_llm_judge is False
    assert caps.ship_grade_judge is False
    assert any("Fail-closed" in note for note in caps.notes)


def test_legacy_provider_trust_env_is_explicit_and_auditable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMEGAPROMPT_TRUST_LEGACY_PROVIDERS", "1")

    class LegacyProvider:
        name = "legacy-contract-provider"

    caps = provider_capabilities(LegacyProvider())

    assert caps.experimental is False
    assert caps.supports_strict_schema is True
    assert caps.supports_llm_judge is True
    assert caps.ship_grade_judge is True
    assert any("OMEGAPROMPT_TRUST_LEGACY_PROVIDERS=1" in note for note in caps.notes)


def test_gemini_is_implemented_target_adapter_but_not_ship_grade_judge() -> None:
    provider = GeminiProvider(model="gemini-test", client=MagicMock())
    caps = provider.capabilities()

    assert caps.placeholder is False
    assert caps.supports_json_object is True
    assert caps.supports_strict_schema is True
    assert caps.supports_llm_judge is True
    assert caps.ship_grade_judge is False
    assert any("ship_grade_judge remains false" in note for note in caps.notes)


def test_local_adapters_are_experimental_targets_not_ship_grade_judges() -> None:
    for provider_name in ["local", "ollama", "vllm", "llama_cpp"]:
        provider = make_provider(
            provider_name,
            client=MagicMock(),
            base_url="http://localhost:11434/v1",
        )
        caps = provider.capabilities()
        assert caps.tier == CapabilityTier.LOCAL
        assert caps.experimental is True
        assert caps.supports_json_object is True
        assert caps.supports_strict_schema is False
        assert caps.supports_llm_judge is False
        assert caps.ship_grade_judge is False


def test_guarded_mode_fails_closed_when_strict_schema_would_degrade() -> None:
    gemini = GeminiProvider(
        model="gemini-test",
        client=MagicMock(),
        native_strict_schema=False,
    )
    with pytest.raises(ProviderError, match="Guarded mode"):
        gemini.call(
            _request(
                response_schema_mode=ResponseSchemaMode.STRICT_SCHEMA,
                output_schema=JudgeResult,
            )
        )

    local = make_provider(
        "ollama",
        client=MagicMock(),
        base_url="http://localhost:11434/v1",
    )
    with pytest.raises(ProviderError, match="does not provide ship-grade strict schema"):
        local.call(
            _request(
                response_schema_mode=ResponseSchemaMode.STRICT_SCHEMA,
                output_schema=JudgeResult,
            )
        )


def test_expedition_mode_records_strict_schema_fallback_degradation() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _openai_target_response(
        '{"scores":{"quality":4},"gate_results":{"correctness":true},"notes":"ok"}'
    )
    provider = make_provider(
        "ollama",
        client=client,
        base_url="http://localhost:11434/v1",
    )

    response = provider.call(
        _request(
            response_schema_mode=ResponseSchemaMode.STRICT_SCHEMA,
            output_schema=JudgeResult,
            execution_profile=ExecutionProfile.EXPEDITION,
        )
    )

    assert isinstance(response.parsed, JudgeResult)
    assert response.degraded_capabilities
    event = response.degraded_capabilities[-1]
    assert event.capability == "structured_output"
    assert event.requested == "strict_schema"
    assert event.applied == "json_object_parse"
    assert event.affects_guarded_boundary is True


def test_openai_compatible_endpoint_downgrades_and_records_parse_fallback() -> None:
    client = MagicMock()
    client.beta.chat.completions.parse.side_effect = RuntimeError(
        "response_format json_schema unsupported"
    )
    client.chat.completions.create.return_value = _openai_target_response(
        '{"scores":{"quality":4},"gate_results":{"correctness":true},"notes":"ok"}'
    )
    provider = OpenAIProvider(
        model="local-model",
        client=client,
        base_url="http://localhost:8000/v1",
    )

    caps = provider.capabilities()
    assert caps.tier == CapabilityTier.LOCAL
    assert caps.supports_strict_schema is False
    assert caps.ship_grade_judge is False
    assert caps.experimental is True

    response = provider.call(
        _request(
            response_schema_mode=ResponseSchemaMode.STRICT_SCHEMA,
            output_schema=JudgeResult,
            execution_profile=ExecutionProfile.EXPEDITION,
        )
    )

    assert isinstance(response.parsed, JudgeResult)
    assert response.usage["input_tokens"] == 13
    assert response.usage["output_tokens"] == 5
    assert any(
        event.capability == "structured_output"
        and event.applied == "json_object_parse"
        for event in response.degraded_capabilities
    )


def test_usage_and_degradation_propagate_to_eval_result_and_artifact() -> None:
    target_event = CapabilityEvent(
        capability="structured_output",
        requested="json_object",
        applied="prompt_only_json",
        reason="backend rejected response_format=json_object",
        user_visible_note="Target provider used prompt-only JSON.",
    )
    judge_event = CapabilityEvent(
        capability="strict_schema",
        requested="strict_schema",
        applied="json_object_parse",
        reason="judge endpoint lacks native parse support",
        user_visible_note="Judge provider used JSON parsing fallback.",
    )

    class FakeTargetProvider:
        name = "ollama"
        model = "contract-target"

        def call(self, request: ProviderRequest) -> ProviderResponse:
            return ProviderResponse(
                text="A substantive answer.",
                usage=normalize_usage(
                    SimpleNamespace(prompt_token_count=11, candidates_token_count=4)
                ),
                degraded_capabilities=[target_event],
            )

        def capabilities(self) -> ProviderCapabilities:
            return ProviderCapabilities(
                provider=self.name,
                tier=CapabilityTier.LOCAL,
                supports_json_object=True,
                experimental=True,
                ship_grade_judge=False,
            )

    class FakeJudge:
        name = "contract-judge"

        def score(self, *, rubric: JudgeRubric, item: object, target_response: str) -> JudgeOutcome:
            return JudgeOutcome(
                result=JudgeResult(
                    scores={"quality": 4},
                    gate_results={"correctness": True},
                    notes="ok",
                ),
                usage=normalize_usage(
                    SimpleNamespace(prompt_tokens=7, completion_tokens=3)
                ),
                degraded_capabilities=[judge_event],
            )

    target = FakeTargetProvider()
    result = PromptTarget(
        target_provider=target,
        judge=FakeJudge(),
        dataset=Dataset.from_items([{"id": "case-1", "input": "x", "reference": "y"}]),
        rubric=_rubric(),
        variants=PromptVariants(system_prompts=["Be precise."]),
    ).evaluate({})

    assert result.usage_summary["input_tokens"] == 18
    assert result.usage_summary["output_tokens"] == 7
    assert {event.capability for event in result.degraded_capabilities} == {
        "structured_output",
        "strict_schema",
    }
    assert result.within_guarded_boundaries is False

    artifact = CalibrationArtifact(
        method="provider-contract",
        unlock_k=0,
        best_params={},
        best_fitness=result.fitness,
        hard_gate_pass_rate=result.hard_gate_pass_rate,
        n_candidates_evaluated=1,
        total_api_calls=2,
        usage_summary=result.usage_summary,
        degraded_capabilities=result.degraded_capabilities,
        target_provider=target.name,
        target_model=target.model,
        target_capabilities=target.capabilities(),
        judge_provider="gemini",
        judge_model="contract-judge",
        judge_capabilities=ProviderCapabilities(
            provider="gemini",
            tier=CapabilityTier.CLOUD,
            supports_json_object=True,
            supports_strict_schema=True,
            supports_llm_judge=True,
            ship_grade_judge=False,
        ),
    )

    payload = artifact.model_dump(mode="json")
    assert payload["usage_summary"] == result.usage_summary
    assert payload["degraded_capabilities"][0]["capability"] == "structured_output"
    assert payload["degraded_capabilities"][1]["capability"] == "strict_schema"
    assert "strict_schema" in artifact.model_dump_json()


def test_provider_capability_docs_are_checked_against_code_contract() -> None:
    docs = (REPO_ROOT / "docs" / "provider-capabilities.md").read_text(encoding="utf-8")

    assert "| Anthropic | Tier 2 cloud-grade | First-class | Ship-grade |" in docs
    assert "| OpenAI | Tier 2 cloud-grade | First-class | Ship-grade |" in docs
    assert (
        "| Gemini | Tier 2 cloud-grade | Implemented target adapter | "
        "Not ship-grade judge |"
    ) in docs
    assert (
        "| `local` / `ollama` / `vllm` / `llama_cpp` | Tier 3 local | "
        "Experimental target path | Exploration-grade judge only |"
    ) in docs
    assert "`placeholder=False` in code" in docs
    assert "ship_grade_judge=False" in docs
    assert "cloud-equivalent judge semantics" in docs

    gemini_caps = GeminiProvider(model="gemini-test", client=MagicMock()).capabilities()
    assert gemini_caps.placeholder is False
    assert gemini_caps.ship_grade_judge is False

    local_caps = make_provider(
        "local",
        client=MagicMock(),
        base_url="http://localhost:11434/v1",
    ).capabilities()
    assert local_caps.experimental is True
    assert local_caps.ship_grade_judge is False

    local_summary = next(line for line in docs.splitlines() if line.startswith("| `local`"))
    assert "Ship-grade" not in local_summary


def test_provider_docs_do_not_market_local_or_gemini_judges_as_ship_grade() -> None:
    docs = (REPO_ROOT / "docs" / "provider-capabilities.md").read_text(encoding="utf-8")

    assert "Gemini as primary LLM judge without explicit expedition boundary crossing" in docs
    assert "local primary LLM judge without explicit expedition boundary crossing" in docs
    assert "parity with frontier cloud models" in docs

    for provider in [
        GeminiProvider(model="gemini-test", client=MagicMock()),
        make_provider("local", client=MagicMock(), base_url="http://localhost:11434/v1"),
        make_provider("ollama", client=MagicMock(), base_url="http://localhost:11434/v1"),
        make_provider("vllm", client=MagicMock(), base_url="http://localhost:8000/v1"),
        make_provider("llama_cpp", client=MagicMock(), base_url="http://localhost:8080/v1"),
    ]:
        assert provider.capabilities().ship_grade_judge is False


def test_cloud_provider_classes_keep_declared_ship_grade_judge_contract() -> None:
    for provider in [
        AnthropicProvider(model="claude-contract", client=MagicMock()),
        OpenAIProvider(model="gpt-contract", client=MagicMock()),
    ]:
        caps = provider.capabilities()
        assert caps.tier == CapabilityTier.CLOUD
        assert caps.supports_strict_schema is True
        assert caps.supports_llm_judge is True
        assert caps.ship_grade_judge is True
