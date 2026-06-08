"""T6 (rescore_count median-of-N wiring) + T5-part1 (EXPEDITION min_kc4).

These cover the omegaprompt (lane B) slice of the threshold recalibration:

T6 — ``rescore_count`` is threaded from the AdaptationPlan through
``apply_adaptation_plan`` into ``PromptTarget`` per-item scoring. When
``rescore_count > 1`` each item's FIXED target response is scored by N
independent judge calls and the lower-median result (by weighted_score)
is kept. Target cost is unchanged (one target call per item); only judge
cost scales with N. Usage sums across all N judge calls and degradation
events are unioned.

T5-part1 — EXPEDITION profile ``default_min_kc4`` raised 0.3 -> 0.5.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from omegaprompt.domain.dataset import Dataset
from omegaprompt.domain.judge import Dimension, HardGate, JudgeResult, JudgeRubric
from omegaprompt.domain.params import PromptVariants
from omegaprompt.domain.profiles import ExecutionProfile
from omegaprompt.core.profiles import policy_for
from omegaprompt.preflight.adaptation import (
    AdaptationPlan,
    apply_adaptation_plan,
)
from omegaprompt.providers.base import ProviderResponse
from omegaprompt.targets.prompt_target import PromptTarget, _select_median_result


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _rubric() -> JudgeRubric:
    return JudgeRubric(
        dimensions=[Dimension(name="q", description="q", weight=1.0, scale=(0, 5))],
        hard_gates=[HardGate(name="g", description="g")],
    )


def _dataset(n: int = 1) -> Dataset:
    return Dataset.from_items([{"id": f"t{i}", "input": f"in{i}"} for i in range(n)])


def _variants() -> PromptVariants:
    return PromptVariants(
        system_prompts=["sp-A"],
        few_shot_examples=[{"input": "q1", "output": "a1"}],
    )


def _target_provider(text: str = "ok"):
    p = MagicMock()
    p.name = "anthropic"
    p.model = "target-model"
    p.call.return_value = ProviderResponse(
        text=text,
        usage={
            "input_tokens": 5,
            "output_tokens": 3,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    )
    return p


def _judge_with_scores(scores: list[int]):
    """A judge whose successive ``score()`` calls return the given per-call
    dimension scores in order (cycling per item via side_effect list).

    Each tuple carries a small fixed usage so cost accumulation is testable.
    """
    judge = MagicMock()
    judge.name = "mock-judge"
    judge.score.side_effect = [
        (
            JudgeResult(scores={"q": s}, gate_results={"g": True}),
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        )
        for s in scores
    ]
    return judge


# --------------------------------------------------------------------------
# _select_median_result unit tests (lower-median semantics)
# --------------------------------------------------------------------------


def _r(score: int) -> JudgeResult:
    return JudgeResult(scores={"q": score}, gate_results={"g": True})


def test_select_median_single_result_returned_verbatim():
    only = _r(3)
    assert _select_median_result([only], _rubric()) is only


def test_select_median_odd_n_picks_middle():
    rubric = _rubric()
    results = [_r(1), _r(5), _r(3)]  # unsorted on purpose
    sel = _select_median_result(results, rubric)
    assert sel.scores["q"] == 3


def test_select_median_even_n_picks_lower_of_two():
    # n=2 -> index (2-1)//2 == 0 of ascending sort -> the lower score.
    rubric = _rubric()
    sel = _select_median_result([_r(4), _r(2)], rubric)
    assert sel.scores["q"] == 2


def test_select_median_empty_raises():
    with pytest.raises(ValueError, match="at least one result"):
        _select_median_result([], _rubric())


# --------------------------------------------------------------------------
# PromptTarget rescore_count behaviour
# --------------------------------------------------------------------------


def test_rescore_default_is_one_and_single_judge_call_per_item():
    """Default rescore_count=1 must be byte-identical to a single judge
    call per item (no behaviour change for existing callers)."""
    target = _target_provider()
    judge = _judge_with_scores([4])  # one item, one call
    t = PromptTarget(
        target_provider=target,
        judge=judge,
        dataset=_dataset(n=1),
        rubric=_rubric(),
        variants=_variants(),
    )
    assert t.rescore_count == 1
    result = t.evaluate({})
    assert target.call.call_count == 1
    assert judge.score.call_count == 1
    assert result.metadata["target_calls"] == 1
    assert result.metadata["judge_calls"] == 1


def test_rescore_three_takes_median_target_called_once_judge_thrice():
    """The cost model + median selection together: for rescore_count=3 on a
    single item the target is called ONCE and the judge THREE times, and the
    kept per-item score is the median of the three judge scores."""
    target = _target_provider()
    # Per-call dimension scores 1, 5, 3 -> median dimension score 3.
    judge = _judge_with_scores([1, 5, 3])
    t = PromptTarget(
        target_provider=target,
        judge=judge,
        dataset=_dataset(n=1),
        rubric=_rubric(),
        variants=_variants(),
        rescore_count=3,
    )
    result = t.evaluate({})

    # (c) target called exactly once per item.
    assert target.call.call_count == 1
    assert result.metadata["target_calls"] == 1
    # (b) judge called N times per item.
    assert judge.score.call_count == 3
    assert result.metadata["judge_calls"] == 3
    # (a) the kept result is the median (dimension score 3 on a 0..5 scale
    #     -> soft_score 3/5 == 0.6).
    per_item = result.metadata["per_item_scores"]
    assert len(per_item) == 1
    assert abs(per_item[0]["soft_score"] - 0.6) < 1e-9


def test_rescore_usage_sums_across_all_judge_calls():
    """The artifact's cost must reflect ALL N judge calls, not one. Each
    judge call reports input_tokens=10/output_tokens=5; three calls -> 30/15
    for the judge, plus the single target call's 5/3."""
    target = _target_provider()
    judge = _judge_with_scores([2, 2, 2])
    t = PromptTarget(
        target_provider=target,
        judge=judge,
        dataset=_dataset(n=1),
        rubric=_rubric(),
        variants=_variants(),
        rescore_count=3,
    )
    result = t.evaluate({})
    usage = result.metadata["usage"]
    # target 5 + 3*judge 10 = 35 input; target 3 + 3*judge 5 = 18 output.
    assert usage["input_tokens"] == 5 + 3 * 10
    assert usage["output_tokens"] == 3 + 3 * 5


def test_rescore_total_api_calls_counts_every_judge_replay():
    target = _target_provider()
    judge = _judge_with_scores([1, 2])  # rescore_count=2, one item
    t = PromptTarget(
        target_provider=target,
        judge=judge,
        dataset=_dataset(n=1),
        rubric=_rubric(),
        variants=_variants(),
        rescore_count=2,
    )
    t.evaluate({})
    # 1 target + 2 judge = 3 provider calls accounted for.
    assert t.total_api_calls == 3


def test_rescore_count_floored_at_one():
    """A nonsensical rescore_count (0 or negative) is clamped to 1, never
    skipping the judge entirely."""
    t = PromptTarget(
        target_provider=_target_provider(),
        judge=_judge_with_scores([1]),
        dataset=_dataset(n=1),
        rubric=_rubric(),
        variants=_variants(),
        rescore_count=0,
    )
    assert t.rescore_count == 1


# --------------------------------------------------------------------------
# apply_adaptation_plan threads rescore_count (monotone max invariant)
# --------------------------------------------------------------------------


def test_apply_plan_returns_rescore_count_fourth():
    plan = AdaptationPlan(rescore_count=3)
    kc4, gap, k, rescore = apply_adaptation_plan(
        plan, min_kc4=0.5, max_gap=0.25, unlock_k=3, rescore_count=1
    )
    assert (kc4, gap, k) == (0.5, 0.25, 3)
    assert rescore == 3


def test_apply_plan_rescore_never_below_caller_default():
    """Monotone invariant: the plan can only RAISE rescore_count, never
    drop it below the caller's default."""
    plan = AdaptationPlan(rescore_count=1)  # plan asks for fewer
    _kc4, _gap, _k, rescore = apply_adaptation_plan(
        plan, min_kc4=0.5, max_gap=0.25, unlock_k=3, rescore_count=2
    )
    assert rescore == 2  # caller's higher value preserved


def test_apply_plan_rescore_defaults_to_one_when_omitted():
    """Backward-compat: callers that don't pass rescore_count get 1 (and a
    plan default of 1 keeps it at 1)."""
    plan = AdaptationPlan()  # rescore_count defaults to 1
    _kc4, _gap, _k, rescore = apply_adaptation_plan(
        plan, min_kc4=0.5, max_gap=0.25, unlock_k=3
    )
    assert rescore == 1


# --------------------------------------------------------------------------
# T5-part1: EXPEDITION profile default_min_kc4 raised 0.3 -> 0.5
# --------------------------------------------------------------------------


def test_expedition_min_kc4_raised_to_half():
    assert policy_for(ExecutionProfile.EXPEDITION).default_min_kc4 == 0.5


def test_guarded_min_kc4_stays_frozen_at_half():
    # GUARDED is explicitly NOT touched by T5-part1; it stays at 0.5.
    assert policy_for(ExecutionProfile.GUARDED).default_min_kc4 == 0.5
