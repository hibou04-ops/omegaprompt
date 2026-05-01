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


def test_prompt_target_rejects_user_supplied_space_with_oversized_idx_max():
    """Reviewer P1 #10: a user-supplied space whose system_prompt_idx_max
    exceeds the variant pool would IndexError mid-eval. PromptTarget must
    reject at construction time."""
    space = MetaAxisSpace(system_prompt_idx_max=99)  # variants has 2 prompts
    with pytest.raises(ValueError, match="system_prompt_idx_max"):
        PromptTarget(
            target_provider=_target_provider(),
            judge=_judge_mock(),
            dataset=_dataset(),
            rubric=_rubric(),
            variants=_variants(),
            space=space,
        )


def test_prompt_target_rejects_user_supplied_space_with_oversized_few_shot_max():
    space = MetaAxisSpace(system_prompt_idx_max=1, few_shot_max=10)  # only 1 shot
    with pytest.raises(ValueError, match="few_shot_max"):
        PromptTarget(
            target_provider=_target_provider(),
            judge=_judge_mock(),
            dataset=_dataset(),
            rubric=_rubric(),
            variants=_variants(),
            space=space,
        )


def test_prompt_target_auto_derived_space_remains_consistent():
    """When space is None PromptTarget builds one from variants.
    The auto-derived bounds must satisfy the cross-validator (regression
    guard against future changes that desynchronise the two)."""
    target = PromptTarget(
        target_provider=_target_provider(),
        judge=_judge_mock(),
        dataset=_dataset(),
        rubric=_rubric(),
        variants=_variants(),
        space=None,
    )
    assert target.space.system_prompt_idx_max == 1  # 2 prompts
    assert target.space.few_shot_max <= 1  # 1 shot in _variants()


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
        "system_prompt_variant",
        "few_shot_count",
        "reasoning_profile",
        "output_budget_bucket",
        "response_schema_mode",
        "tool_policy_variant",
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
    result = t.evaluate({"system_prompt_variant": 1, "few_shot_count": 1, "reasoning_profile": 2})

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
    assert resolved["system_prompt_variant"] == 0
    assert resolved["few_shot_count"] == 0


def test_evaluate_clamps_out_of_range_params_under_expedition():
    """Reviewer P1 #11: under expedition profile out-of-range values
    are clamped, but each drift emits a BoundaryWarning so the artifact
    can audit which optimizer outputs were ignored."""
    from omegaprompt.domain.profiles import ExecutionProfile

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
        execution_profile=ExecutionProfile.EXPEDITION,
    )
    result = t.evaluate({
        "system_prompt_variant": 99,
        "few_shot_count": -10,
        "reasoning_profile": 999,
        "output_budget_bucket": -999,
    })
    resolved = result.metadata["resolved_params"]
    assert resolved["system_prompt_variant"] == 1
    assert resolved["few_shot_count"] == 0

    # Each clamped axis emits a separate BoundaryWarning.
    clamp_codes = [
        bw.code for bw in result.boundary_warnings if bw.code == "param_clamped"
    ]
    assert len(clamp_codes) == 4  # sys, fs, reasoning, budget
    summaries = " | ".join(bw.summary for bw in result.boundary_warnings)
    assert "system_prompt_variant" in summaries
    assert "few_shot_count" in summaries

    # Param drift moves us off the guarded path.
    assert result.within_guarded_boundaries is False


def test_evaluate_raises_under_guarded_when_param_out_of_range():
    """Reviewer P1 #11: silent clamp is dangerous in audit-first mode.
    Under guarded profile (the default) an out-of-range value is a
    setup bug that must surface before the eval runs, not be papered
    over with a clamped artifact."""
    target = _target_provider()
    judge = _judge_mock()

    t = PromptTarget(
        target_provider=target,
        judge=judge,
        dataset=_dataset(n=1),
        rubric=_rubric(),
        variants=_variants(),
    )
    with pytest.raises(ValueError, match="out of axis bounds"):
        t.evaluate({"system_prompt_variant": 99})


def test_evaluate_in_range_params_pass_under_guarded():
    """Sanity: legitimate params (no clamping needed) work under guarded."""
    target = _target_provider()
    judge = _judge_mock(score=1, gate=True)

    t = PromptTarget(
        target_provider=target,
        judge=judge,
        dataset=_dataset(n=1),
        rubric=_rubric(),
        variants=_variants(),
    )
    result = t.evaluate({"system_prompt_variant": 1, "few_shot_count": 0})
    assert result.boundary_warnings == []
    assert result.within_guarded_boundaries is True


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
    t.evaluate({"reasoning_profile": 2, "output_budget_bucket": 2})
    request = target.call.call_args.args[0]
    # reasoning_profiles default = [OFF, STANDARD, DEEP], idx=2 -> DEEP
    assert request.reasoning_profile == ReasoningProfile.DEEP
    # output_budgets default = [SMALL, MEDIUM, LARGE], idx=2 -> LARGE
    assert request.output_budget_bucket == OutputBudgetBucket.LARGE


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
