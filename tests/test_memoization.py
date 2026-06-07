"""H1 memoization gate: PromptTarget.evaluate() caches by resolved params.

H1 is a pure reduction behind the public surface: a repeat evaluate() of an
already-seen RESOLVED configuration returns the prior EvalResult without
re-calling the providers. These tests are the call-count GATE for H1 — if the
cache does not hit, H1 is a harmless no-op and these assertions catch it.

Zero network: a counting target provider + counting judge stand in for live
calls so we can assert exact call counts.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pytest

from omegaprompt.domain.dataset import Dataset
from omegaprompt.domain.judge import Dimension, HardGate, JudgeResult, JudgeRubric
from omegaprompt.domain.params import PromptVariants
from omegaprompt.providers.base import (
    CapabilityTier,
    ProviderCapabilities,
    ProviderResponse,
)
from omegaprompt.targets.prompt_target import PromptTarget


def _rubric() -> JudgeRubric:
    return JudgeRubric(
        dimensions=[Dimension(name="q", description="q", weight=1.0, scale=(0, 1))],
        hard_gates=[HardGate(name="g", description="g")],
    )


def _dataset(n: int = 3) -> Dataset:
    return Dataset.from_items([{"id": f"t{i}", "input": f"in{i}"} for i in range(n)])


def _variants() -> PromptVariants:
    return PromptVariants(
        system_prompts=["sp-A", "sp-B"],
        few_shot_examples=[{"input": "q1", "output": "a1"}],
    )


class _CountingProvider:
    name = "anthropic"
    model = "target-model"

    def __init__(self) -> None:
        self.calls = 0

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider=self.name,
            tier=CapabilityTier.CLOUD,
            supports_strict_schema=True,
            supports_json_object=True,
            supports_reasoning_profiles=True,
            supports_usage_accounting=True,
            supports_llm_judge=True,
            ship_grade_judge=True,
        )

    def call(self, request) -> ProviderResponse:
        self.calls += 1
        return ProviderResponse(
            text="ok",
            usage={
                "input_tokens": 5,
                "output_tokens": 3,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        )


class _CountingJudge:
    name = "mock-judge"

    def __init__(self) -> None:
        self.calls = 0

    def score(self, *, rubric, item, target_response):
        self.calls += 1
        return (
            JudgeResult(scores={"q": 1}, gate_results={"g": True}),
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        )


def test_second_evaluate_of_same_resolved_params_adds_zero_calls():
    """The core H1 gate. A repeat evaluate() of the same resolved config hits
    the cache: zero new provider/judge calls, no total_api_calls increment, and
    the SAME EvalResult object returned."""
    target = _CountingProvider()
    judge = _CountingJudge()
    t = PromptTarget(
        target_provider=target,
        judge=judge,
        dataset=_dataset(n=3),
        rubric=_rubric(),
        variants=_variants(),
    )

    first = t.evaluate({"system_prompt_variant": 1, "few_shot_count": 1})
    assert target.calls == 3
    assert judge.calls == 3
    assert t.total_api_calls == 6

    second = t.evaluate({"system_prompt_variant": 1, "few_shot_count": 1})
    # GATE: the cache hit performs no work.
    assert target.calls == 3, "H1 cache did not hit: target was re-called"
    assert judge.calls == 3, "H1 cache did not hit: judge was re-called"
    assert t.total_api_calls == 6, "H1 cache hit must not increment total_api_calls"
    assert second is first, "cache must return the prior EvalResult verbatim"


def test_cache_keys_on_resolved_not_raw_params():
    """Two different RAW param dicts that resolve to the same configuration
    share one cache entry (the key is the resolved params)."""
    target = _CountingProvider()
    judge = _CountingJudge()
    t = PromptTarget(
        target_provider=target,
        judge=judge,
        dataset=_dataset(n=2),
        rubric=_rubric(),
        variants=_variants(),
    )

    # Raw {} resolves to the neutral configuration. An explicit dict that names
    # every neutral value resolves to the SAME ResolvedPromptParams.
    t.evaluate({})
    assert target.calls == 2
    neutral = t.neutral_params()
    t.evaluate(dict(neutral))
    # Same resolved config -> cache hit -> no new calls.
    assert target.calls == 2, "raw vs explicit-neutral params should share a cache entry"
    assert judge.calls == 2


def test_distinct_resolved_params_do_not_share_cache():
    """A genuinely new resolved configuration is computed (not a false hit)."""
    target = _CountingProvider()
    judge = _CountingJudge()
    t = PromptTarget(
        target_provider=target,
        judge=judge,
        dataset=_dataset(n=2),
        rubric=_rubric(),
        variants=_variants(),
    )
    t.evaluate({"system_prompt_variant": 0})
    assert target.calls == 2
    t.evaluate({"system_prompt_variant": 1})
    assert target.calls == 4, "a new resolved config must run fresh"


def test_history_and_idempotence_unaffected_by_cache():
    """H1-b: unique_param_count() and best_guarded_eval() are correct whether or
    not the duplicate eval was served from cache. The cache short-circuits the
    second history append, but the deduped/max consumers are idempotent."""
    target = _CountingProvider()
    judge = _CountingJudge()
    t = PromptTarget(
        target_provider=target,
        judge=judge,
        dataset=_dataset(n=2),
        rubric=_rubric(),
        variants=_variants(),
    )
    t.evaluate({"system_prompt_variant": 0})
    t.evaluate({"system_prompt_variant": 0})  # cache hit
    t.evaluate({"system_prompt_variant": 1})
    # Only two distinct resolved configs were actually computed.
    assert t.unique_param_count() == 2
    best = t.best_guarded_eval()
    assert best is not None
    assert best.within_guarded_boundaries is True


# ----------------------------------------------------------------------
# Integration form: the real H1 gate against a full mocked calibrate()
# ----------------------------------------------------------------------


def _write_calibrate_fixtures(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    train = tmp_path / "train.jsonl"
    train.write_text(
        "\n".join(json.dumps({"id": f"t{i}", "input": f"task {i}"}) for i in range(4)) + "\n",
        encoding="utf-8",
    )
    test = tmp_path / "test.jsonl"
    test.write_text(
        "\n".join(json.dumps({"id": f"t{i}", "input": f"task {i} test"}) for i in range(3)) + "\n",
        encoding="utf-8",
    )
    rubric = {
        "dimensions": [{"name": "accuracy", "description": "r", "weight": 1.0}],
        "hard_gates": [{"name": "no_refusal", "description": "a", "evaluator": "judge"}],
    }
    variants = {
        "system_prompts": [
            "You are a helpful assistant.",
            "You are a terse senior engineer.",
        ],
        "few_shot_examples": [{"input": "1+1=", "output": "2"}],
    }
    return train, test, rubric, variants


def test_calibrate_best_candidate_eval_adds_zero_calls(monkeypatch, tmp_path):
    """The real H1 gate, isolated to runtime.py's best-candidate re-eval.

    Call order on ``train_target`` inside calibrate():
        neutral eval -> run_p1 grid (many evals) -> best-candidate eval
        (runtime.py best_train_eval = train_target.evaluate(best_candidate))
        -> best_guarded_eval() (no evaluate call).
    The test_target is a separate instance with its own cache. So the LAST
    ``train_target.evaluate()`` call is exactly the best-candidate re-eval — we
    assert that specific call is a cache HIT (zero provider calls), not merely
    that the cache hit somewhere. If it is a miss, H1 saved nothing at the gate
    the task cares about and this fails LOUD."""
    import omegaprompt.runtime as rt
    from omegaprompt.judges import llm_judge as llm_judge_mod
    from omegaprompt.targets.prompt_target import PromptTarget as _PT

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")

    def fake_make_provider(name, model=None, api_key=None, base_url=None, **_):
        return _CountingProvider()

    monkeypatch.setattr("omegaprompt.runtime.make_provider", fake_make_provider)

    def fake_score(self, *, rubric, item, target_response):
        # Deterministic by params so run_p1 has a clean signal.
        score = 1 + min(4, len(target_response) % 5)
        return (
            JudgeResult(scores={"accuracy": score}, gate_results={"no_refusal": True}, notes="x"),
            {
                "input_tokens": 20,
                "output_tokens": 10,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        )

    monkeypatch.setattr(llm_judge_mod.LLMJudge, "score", fake_score)

    # Record (instance id, was_hit) for every PromptTarget.evaluate() call.
    events: list[tuple[int, bool]] = []
    orig_evaluate = _PT.evaluate

    def counting_evaluate(self, params):
        resolved, _ = self._resolve_params(params)
        key = json.dumps(
            resolved.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        events.append((id(self), key in self._eval_cache))
        return orig_evaluate(self, params)

    monkeypatch.setattr(_PT, "evaluate", counting_evaluate)

    train, test, rubric, variants = _write_calibrate_fixtures(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        artifact = rt.calibrate(
            train,
            rubric=rubric,
            variants=variants,
            target="anthropic",
            judge="anthropic",
            test=test,
            tuning=rt.CalibrateTuning(unlock_k=1, max_gap=0.99, min_kc4=-1.0),
        )

    assert artifact.total_api_calls > 0
    # The first evaluate() is on train_target (neutral). Every subsequent
    # train_target evaluate shares that instance id; the LAST such call is the
    # best-candidate re-eval (runtime.py best_train_eval).
    train_target_id = events[0][0]
    train_events = [hit for (inst, hit) in events if inst == train_target_id]
    assert len(train_events) >= 2  # at least neutral + best-candidate
    assert train_events[-1] is True, (
        "H1 GATE FAILED: the best-candidate re-eval (last train_target.evaluate, "
        "runtime.py) was a cache MISS — it re-called the providers. H1 is a no-op "
        "(still safe, but it saved no calls); investigate before claiming savings."
    )
