"""Reviewer P0: judge provider degradation must propagate to artifact.

Pre-fix flow:

  judge_provider.call() -> ProviderResponse with degraded_capabilities
  LLMJudge.score() -> drops degraded_capabilities, returns (result, usage)
  PromptTarget.evaluate() -> only accumulates target_response degradation
  EvalItemResult / EvalResult / CalibrationArtifact -> judge degradation invisible

Post-fix:

- ``Judge.score()`` returns a ``JudgeOutcome`` carrying
  ``degraded_capabilities`` + ``latency_ms`` from the underlying provider.
- ``JudgeOutcome.__iter__`` yields ``(result, usage)`` so existing
  tuple-unpacking callers keep working.
- ``PromptTarget.evaluate()`` reads ``judge_outcome.degraded_capabilities``
  and merges them into both per-item and aggregate degradation.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from omegaprompt.domain.dataset import DatasetItem
from omegaprompt.domain.judge import Dimension, HardGate, JudgeResult, JudgeRubric
from omegaprompt.judges import LLMJudge, RuleJudge
from omegaprompt.judges.base import JudgeOutcome
from omegaprompt.judges.ensemble_judge import EnsembleJudge
from omegaprompt.judges.rule_judge import default_no_refusal
from omegaprompt.providers.base import (
    CapabilityEvent,
    CapabilityTier,
    ProviderCapabilities,
    ProviderResponse,
)


def _rubric() -> JudgeRubric:
    return JudgeRubric(
        dimensions=[Dimension(name="acc", description="x", weight=1.0)],
        hard_gates=[
            HardGate(name="correctness", description="c", evaluator="judge"),
        ],
    )


def _provider_with_degradation() -> MagicMock:
    """Returns a ProviderResponse carrying a CapabilityEvent."""
    event = CapabilityEvent(
        capability="strict_schema",
        requested="strict_schema",
        applied="json_object_parse",
        reason="adapter_unsupported",
        user_visible_note="judge fell back from STRICT_SCHEMA to JSON_OBJECT",
    )
    provider = MagicMock()
    provider.name = "openai"
    provider.model = "gpt-4o"
    provider.call.return_value = ProviderResponse(
        parsed=JudgeResult(
            scores={"acc": 4},
            gate_results={"correctness": True},
        ),
        usage={
            "input_tokens": 100, "output_tokens": 30,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        },
        latency_ms=120.0,
        degraded_capabilities=[event],
    )
    provider.capabilities = lambda: ProviderCapabilities(
        provider="openai",
        tier=CapabilityTier.CLOUD,
        supports_strict_schema=True,
        supports_llm_judge=True,
        ship_grade_judge=True,
    )
    return provider


# ---------------------------------------------------------------------------
# JudgeOutcome shape + backward-compat iteration.
# ---------------------------------------------------------------------------


def test_judge_outcome_iterates_as_result_usage_pair():
    outcome = JudgeOutcome(
        result=JudgeResult(scores={"a": 1}, gate_results={}),
        usage={"input_tokens": 10},
    )
    a, b = outcome
    assert a == outcome.result
    assert b == outcome.usage


def test_judge_outcome_carries_degradation_and_latency():
    event = CapabilityEvent(
        capability="x", requested="x", applied="y",
        reason="r", user_visible_note="n",
    )
    outcome = JudgeOutcome(
        result=JudgeResult(scores={"a": 1}, gate_results={}),
        usage={},
        degraded_capabilities=[event],
        latency_ms=50.0,
    )
    assert outcome.degraded_capabilities == [event]
    assert outcome.latency_ms == 50.0


# ---------------------------------------------------------------------------
# LLMJudge propagates degradation from the provider response.
# ---------------------------------------------------------------------------


def test_llm_judge_returns_judge_outcome_with_degradation():
    judge = LLMJudge(provider=_provider_with_degradation())
    outcome = judge.score(
        rubric=_rubric(),
        item=DatasetItem(id="t1", input="x", reference="y"),
        target_response="response",
    )
    assert isinstance(outcome, JudgeOutcome)
    assert len(outcome.degraded_capabilities) == 1
    assert outcome.degraded_capabilities[0].capability == "strict_schema"
    assert outcome.latency_ms == 120.0


def test_llm_judge_outcome_unpacks_for_legacy_callers():
    """``result, usage = judge.score(...)`` still works."""
    judge = LLMJudge(provider=_provider_with_degradation())
    result, usage = judge.score(
        rubric=_rubric(),
        item=DatasetItem(id="t1", input="x"),
        target_response="r",
    )
    assert isinstance(result, JudgeResult)
    assert usage["input_tokens"] == 100


# ---------------------------------------------------------------------------
# RuleJudge returns JudgeOutcome with empty degradation.
# ---------------------------------------------------------------------------


def test_rule_judge_returns_judge_outcome_empty_degradation():
    rubric = JudgeRubric(
        dimensions=[Dimension(name="acc", description="x", weight=1.0)],
        hard_gates=[HardGate(name="no_refusal", description="x", evaluator="rule")],
    )
    judge = RuleJudge(checks=[default_no_refusal()])
    outcome = judge.score(
        rubric=rubric,
        item=DatasetItem(id="t1", input="x"),
        target_response="A real and substantive response.",
    )
    assert isinstance(outcome, JudgeOutcome)
    assert outcome.degraded_capabilities == []
    assert outcome.latency_ms == 0.0


# ---------------------------------------------------------------------------
# EnsembleJudge passes through fallback's degradation.
# ---------------------------------------------------------------------------


def test_ensemble_judge_propagates_fallback_degradation():
    rubric = JudgeRubric(
        dimensions=[Dimension(name="acc", description="x", weight=1.0)],
        hard_gates=[
            HardGate(name="no_refusal", description="x", evaluator="rule"),
            HardGate(name="correctness", description="x", evaluator="judge"),
        ],
    )
    rule = RuleJudge(checks=[default_no_refusal()])
    llm = LLMJudge(provider=_provider_with_degradation())
    ensemble = EnsembleJudge(rule_judge=rule, fallback=llm)
    outcome = ensemble.score(
        rubric=rubric,
        item=DatasetItem(id="t1", input="x"),
        target_response="A real attempt at the input.",
    )
    assert isinstance(outcome, JudgeOutcome)
    # Fallback's degradation surfaces through the ensemble:
    assert len(outcome.degraded_capabilities) == 1
    assert outcome.degraded_capabilities[0].capability == "strict_schema"


# ---------------------------------------------------------------------------
# PromptTarget aggregates judge + target degradation.
# ---------------------------------------------------------------------------


def test_prompt_target_aggregates_judge_degradation_in_eval_result(monkeypatch):
    """The EvalResult.degraded_capabilities should include judge events,
    not just target events. Pre-fix only target events surfaced."""
    from omegaprompt.dataset import Dataset
    from omegaprompt.domain.params import MetaAxisSpace, PromptVariants
    from omegaprompt.targets.prompt_target import PromptTarget

    # Target provider returns clean response, no degradation
    target_provider = MagicMock()
    target_provider.name = "anthropic"
    target_provider.model = "claude"
    target_provider.call.return_value = ProviderResponse(
        text="response",
        usage={
            "input_tokens": 50, "output_tokens": 20,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        },
        latency_ms=10.0,
        degraded_capabilities=[],  # clean target
    )
    target_provider.capabilities = lambda: ProviderCapabilities(
        provider="anthropic",
        tier=CapabilityTier.CLOUD,
        supports_strict_schema=True,
        supports_llm_judge=True,
        ship_grade_judge=True,
    )

    # Judge provider degraded (fell back from strict schema):
    judge = LLMJudge(provider=_provider_with_degradation())

    target = PromptTarget(
        target_provider=target_provider,
        judge=judge,
        dataset=Dataset(items=[DatasetItem(id="t1", input="x", reference="y")]),
        rubric=_rubric(),
        variants=PromptVariants(
            system_prompts=["You are precise."],
            few_shot_examples=[{"input": "1+1", "output": "2"}],
        ),
    )
    result = target.evaluate({})

    # Aggregate degradation includes the judge fallback event:
    capability_names = {e.capability for e in result.degraded_capabilities}
    assert "strict_schema" in capability_names
