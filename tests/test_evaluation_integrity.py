"""Observed regression cases, deterministic costs, and public-contract checks."""
from concurrent.futures import ThreadPoolExecutor as RealExecutor

import pytest

from tests.test_memoization import _CountingProvider, _CountingJudge, _dataset, _rubric, _variants
from omegaprompt.targets.prompt_target import PromptTarget
from omegaprompt.domain.profiles import ExecutionProfile
from omegaprompt.core.walkforward import evaluate_walk_forward
from omegaprompt.preflight.contracts import PreflightReport
from omegaprompt.preflight.adaptation import derive_adaptation_plan


def target(n=3, workers=1):
    return PromptTarget(target_provider=_CountingProvider(), judge=_CountingJudge(),
                        dataset=_dataset(n), rubric=_rubric(), variants=_variants(), max_workers=workers)


@pytest.mark.parametrize("mutation", ["dataset", "rubric", "variants", "profile", "model", "judge", "space"])
def test_public_context_change_invalidates_observation_and_selection(mutation):
    t = target()
    old = t.evaluate({})
    if mutation == "dataset":
        t.dataset.items[0].input = "different task"
    elif mutation == "rubric":
        t.rubric.dimensions[0].scale = (0, 2)
    elif mutation == "variants":
        t.variants.system_prompts[0] = "different prompt"
    elif mutation == "profile":
        t.execution_profile = ExecutionProfile.EXPEDITION
    elif mutation == "model":
        t.target_provider.model = "different-model"
    elif mutation == "judge":
        t.judge = _CountingJudge()
    else:
        t.space.few_shot_max = 0
    new = t.evaluate({})
    assert new is not old
    assert t.target_provider.calls == 6
    assert t.evaluation_history == [new]
    if mutation == "rubric":
        assert new.fitness == .5


def test_fresh_evaluation_has_real_calls_while_replay_does_not():
    t = target()
    first = t.evaluate({})
    assert t.evaluate({}) is first
    assert t.cache_hits == 1 and t.total_api_calls == 6
    fresh = t.evaluate({}, use_cache=False)
    assert fresh is not first and t.total_api_calls == 12
    assert len(t.evaluation_history) == 2
    t.clear_cache()
    assert t.total_api_calls == 12 and t.best_guarded_eval() is None


@pytest.mark.parametrize("change", ["budget", "nested_model", "rule_check"])
def test_builtin_judge_configuration_invalidates_cached_observation(change):
    from omegaprompt.judges import EnsembleJudge, LLMJudge, RuleJudge
    from omegaprompt.judges.rule_judge import RuleCheck
    from omegaprompt.domain.enums import OutputBudgetBucket
    t = target()
    llm = LLMJudge(provider=_CountingProvider())
    llm.score = _CountingJudge().score  # offline deterministic judge observation
    rules = RuleJudge(checks=[RuleCheck("g", lambda *_: True)])
    t.judge = EnsembleJudge(rule_judge=rules, fallback=llm)
    first = t.evaluate({})
    if change == "budget":
        llm.output_budget = OutputBudgetBucket.LARGE
    elif change == "nested_model":
        llm.provider.model = "changed judge model"
    else:
        rules.checks["g"] = RuleCheck("g", lambda *_: False)
    assert t.evaluate({}) is not first
    assert t.target_provider.calls == 6


def test_clamped_params_do_not_replay_a_clean_boundary_verdict():
    t = target()
    t.execution_profile = ExecutionProfile.EXPEDITION
    clean = t.evaluate({"system_prompt_variant": 1})
    clamped = t.evaluate({"system_prompt_variant": 99})
    assert clean.within_guarded_boundaries
    assert not clamped.within_guarded_boundaries
    assert clamped.boundary_warnings and t.total_api_calls == 12


def test_parallel_submission_window_is_bounded_without_dropping_items(monkeypatch):
    counts = {"pending": 0, "peak": 0}
    class TrackingExecutor(RealExecutor):
        def submit(self, *args, **kwargs):
            future = super().submit(*args, **kwargs)
            counts["pending"] += 1
            counts["peak"] = max(counts["peak"], counts["pending"])
            original = future.result
            def result(*a, **kw):
                try:
                    return original(*a, **kw)
                finally:
                    counts["pending"] -= 1
            future.result = result
            return future
    monkeypatch.setattr("omegaprompt.targets.prompt_target.ThreadPoolExecutor", TrackingExecutor)
    t = target(2000, 4)
    result = t.evaluate({})
    assert counts["peak"] == 4
    assert [item.item_id for item in result.item_results] == [f"t{i}" for i in range(2000)]
    assert t.total_api_calls == 4000 and result.n_trials == 2000


@pytest.mark.parametrize("failure", ["target", "judge"])
def test_failed_calls_remain_accounted_without_a_completed_result(failure):
    t = target()
    def boom(*args, **kwargs):
        raise RuntimeError("fixture failure")
    if failure == "target":
        t.target_provider.call = boom
    else:
        t.judge.score = boom
    with pytest.raises(RuntimeError, match="fixture failure"):
        t.evaluate({})
    assert t.total_api_calls == (1 if failure == "target" else 2)
    assert t.last_usage["input_tokens"] == (0 if failure == "target" else 5)
    assert not t.evaluation_history and not t._eval_cache
    assert t.failed_evaluations[0]["api_calls"] == t.total_api_calls


@pytest.mark.parametrize("options", [
    {"train_best_fitness": float("nan")}, {"test_fitness": float("inf")},
    {"max_gap": -1}, {"min_kc4": 2}, {"validation_mode": "typo"},
    {"per_item_train": {"x": float("nan")}},
])
def test_nonfinite_and_invalid_gate_inputs_are_errors(options):
    args = {"train_best_fitness": .8, "test_fitness": .8, **options}
    with pytest.raises(ValueError):
        evaluate_walk_forward(**args)


def test_preflight_measurement_state_round_trip_and_failed_measurement_escalation():
    report = PreflightReport(measurement_status={"judge_consistency": "failed"},
                             probe_accounting={"attempted_calls": 2})
    restored = PreflightReport.model_validate_json(report.model_dump_json())
    assert restored == report
    plan = derive_adaptation_plan(report=restored)
    assert plan.requires_manual_review
    assert not PreflightReport().measurement_status  # legacy unknown, no inferred pass
    with pytest.raises(ValueError):
        PreflightReport(measurement_status={"judge_consistency": "passed"})


@pytest.mark.parametrize("mode,change", [("paired", "content"), ("paired", "ids"), ("disjoint", "ids"), ("disjoint", "content")])
def test_declared_split_contradiction_is_rejected_before_provider_calls(monkeypatch, mode, change):
    from omegaprompt import runtime
    from omegaprompt.domain.dataset import Dataset
    train = _dataset()
    test = train.model_copy(deep=True)
    if (mode == "paired" and change == "content"):
        test.items[0].input = "a different task"
    elif (mode == "paired" and change == "ids") or (mode == "disjoint" and change == "content"):
        for item in test.items:
            item.id += "-renamed"
    monkeypatch.setattr(runtime, "_resolve_provider", lambda *_: pytest.fail("provider resolution must not run"))
    with pytest.raises(ValueError, match="validation_mode"):
        runtime.calibrate(train, test=test, rubric=_rubric(), variants=_variants(),
                          target="anthropic", tuning=runtime.CalibrateTuning(validation_mode=mode))
