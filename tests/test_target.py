"""PromptTarget end-to-end tests with mocked providers + judges."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from omegaprompt.domain.dataset import Dataset
from omegaprompt.domain.enums import OutputBudgetBucket, ReasoningProfile
from omegaprompt.domain.judge import Dimension, HardGate, JudgeResult, JudgeRubric
from omegaprompt.domain.params import MetaAxisSpace, PromptVariants
from omegaprompt.providers.base import ProviderResponse
from omegaprompt.targets.prompt_target import PromptTarget


def _rubric() -> JudgeRubric:
    return JudgeRubric(
        dimensions=[Dimension(name="q", description="q", weight=1.0, scale=(0, 1))],
        hard_gates=[HardGate(name="g", description="g")],
    )


def _dataset(n: int = 2) -> Dataset:
    return Dataset.from_items([{"id": f"t{i}", "input": f"in{i}"} for i in range(n)])


def _variants() -> PromptVariants:
    return PromptVariants(
        system_prompts=["sp-A", "sp-B"],
        few_shot_examples=[{"input": "q1", "output": "a1"}],
    )


def _target_provider(response_text: str = "ok"):
    p = MagicMock()
    p.name = "anthropic"
    p.model = "target-model"
    p.call.return_value = ProviderResponse(
        text=response_text,
        usage={
            "input_tokens": 5,
            "output_tokens": 3,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    )
    return p


def _judge_mock(score: int = 1, gate: bool = True):
    """Mock Judge.score returning fixed values."""
    judge = MagicMock()
    judge.name = "mock-judge"
    judge.score.return_value = (
        JudgeResult(scores={"q": score}, gate_results={"g": gate}),
        {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 100,
        },
    )
    return judge


def test_prompt_target_rejects_empty_variants():
    empty = PromptVariants.model_construct(system_prompts=[], few_shot_examples=[])
    with pytest.raises(ValueError, match="system prompt variant"):
        PromptTarget(
            target_provider=_target_provider(),
            judge=_judge_mock(),
            dataset=_dataset(),
            rubric=_rubric(),
            variants=empty,
        )


def test_prompt_target_rejects_empty_dataset():
    empty_ds = Dataset(items=[])
    with pytest.raises(ValueError, match="non-empty dataset"):
        PromptTarget(
            target_provider=_target_provider(),
            judge=_judge_mock(),
            dataset=empty_ds,
            rubric=_rubric(),
            variants=_variants(),
        )


def test_param_space_returns_meta_axes():
    t = PromptTarget(
        target_provider=_target_provider(),
        judge=_judge_mock(),
        dataset=_dataset(),
        rubric=_rubric(),
        variants=_variants(),
    )
    specs = t.param_space()
    names = {getattr(s, "name", None) for s in specs}
    assert names == {
        "system_prompt_idx",
        "few_shot_count",
        "reasoning_profile_idx",
        "output_budget_idx",
        "response_schema_mode_idx",
        "tool_policy_idx",
    }


def test_evaluate_all_pass_gives_full_fitness():
    target = _target_provider("good response")
    judge = _judge_mock(score=1, gate=True)

    t = PromptTarget(
        target_provider=target,
        judge=judge,
        dataset=_dataset(n=3),
        rubric=_rubric(),
        variants=_variants(),
    )
    result = t.evaluate({"system_prompt_idx": 1, "few_shot_count": 1, "reasoning_profile_idx": 2})

    assert t.total_api_calls == 6  # 3 target + 3 judge
    assert target.call.call_count == 3
    assert judge.score.call_count == 3
    assert abs(result.fitness - 1.0) < 1e-9
    assert result.n_trials == 3
    assert result.metadata["hard_gate_pass_rate"] == 1.0
    assert result.metadata["target_provider"] == "anthropic"
    assert result.metadata["target_model"] == "target-model"
    assert result.metadata["judge_name"] == "mock-judge"


def test_evaluate_gate_failure_zeroes_item():
    target = _target_provider()
    judge = MagicMock()
    judge.name = "mock"
    judge.score.side_effect = [
        (JudgeResult(scores={"q": 1}, gate_results={"g": True}), {}),
        (JudgeResult(scores={"q": 1}, gate_results={"g": False}), {}),
    ]

    t = PromptTarget(
        target_provider=target,
        judge=judge,
        dataset=_dataset(n=2),
        rubric=_rubric(),
        variants=_variants(),
    )
    result = t.evaluate({})
    assert abs(result.fitness - 0.5) < 1e-9
    assert result.metadata["hard_gate_pass_rate"] == 0.5


def test_evaluate_resolves_defaults_for_missing_params():
    target = _target_provider()
    judge = _judge_mock(score=0, gate=True)

    t = PromptTarget(
        target_provider=target,
        judge=judge,
        dataset=_dataset(n=1),
        rubric=_rubric(),
        variants=_variants(),
    )
    result = t.evaluate({})
    resolved = result.metadata["resolved_params"]
    assert resolved["system_prompt_idx"] == 0
    assert resolved["few_shot_count"] == 0


def test_evaluate_clamps_out_of_range_params():
    target = _target_provider()
    judge = _judge_mock(score=1, gate=True)

    t = PromptTarget(
        target_provider=target,
        judge=judge,
        dataset=_dataset(n=1),
        rubric=_rubric(),
        variants=_variants(),
        space=MetaAxisSpace(
            system_prompt_idx_max=1,
            few_shot_min=0,
            few_shot_max=1,
        ),
    )
    result = t.evaluate({
        "system_prompt_idx": 99,
        "few_shot_count": -10,
        "reasoning_profile_idx": 999,
        "output_budget_idx": -999,
    })
    resolved = result.metadata["resolved_params"]
    assert resolved["system_prompt_idx"] == 1
    assert resolved["few_shot_count"] == 0


def test_provider_request_carries_meta_axes():
    target = _target_provider()
    judge = _judge_mock(score=1, gate=True)
    t = PromptTarget(
        target_provider=target,
        judge=judge,
        dataset=_dataset(n=1),
        rubric=_rubric(),
        variants=_variants(),
    )
    t.evaluate({"reasoning_profile_idx": 2, "output_budget_idx": 2})
    request = target.call.call_args.args[0]
    # reasoning_profiles default = [OFF, STANDARD, DEEP], idx=2 -> DEEP
    assert request.reasoning_profile == ReasoningProfile.DEEP
    # output_budgets default = [SMALL, MEDIUM, LARGE], idx=2 -> LARGE
    assert request.output_budget == OutputBudgetBucket.LARGE


def test_usage_accumulates_across_evaluations():
    target = _target_provider()
    judge = _judge_mock()
    t = PromptTarget(
        target_provider=target,
        judge=judge,
        dataset=_dataset(n=2),
        rubric=_rubric(),
        variants=_variants(),
    )
    t.evaluate({})
    first_total = t.last_usage["input_tokens"]
    t.evaluate({})
    assert t.last_usage["input_tokens"] > first_total
    assert t.total_api_calls == 8
