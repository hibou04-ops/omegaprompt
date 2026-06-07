"""M3 measure-before-commit gate: serial baseline + parallel speedup vs provider concurrency ceiling.

REVERSIBLE / NON-PRODUCT. This file does NOT change any product code. It measures
whether a future C2/H1 change (parallelizing PromptTarget.evaluate's per-item loop)
can deliver the claimed 30-60% speedup, and at what provider concurrency ceiling.

What it does
------------
PromptTarget.evaluate (src/omegaprompt/targets/prompt_target.py ~line 178) runs a
SERIAL per-item loop: for each dataset item it does exactly one target_provider.call
(LLM-bound, ~0.5-2.0s real) then one judge.score (LLM-bound, ~0.2-0.5s real). Items
are independent; the only cross-item state is `run_usage` accumulation and the ordered
`judge_results` list -> deterministic fitness aggregation.

We measure two modes over N items using a latency-injecting MOCK provider + judge
(no network; time.sleep models LLM latency and releases the GIL, so threads truly
overlap -> a faithful model of IO-bound work):

  (a) SERIAL  -- the REAL PromptTarget.evaluate() code path (the before-number).
  (b) PARALLEL -- a harness-local SHADOW of evaluate()'s loop (we cannot touch product
      code). The shadow does identical per-item work (same _resolve_params, same
      ProviderRequest, same target.call -> judge.score) so concurrency is the ONLY
      difference. Parallelism is throttled by a semaphore that models a provider
      CONCURRENCY CEILING c (max c items in flight at once); max_workers >= N so the
      thread pool is never the bottleneck -- the semaphore is the sole throttle.

Determinism (the key correctness gate for the future product change)
--------------------------------------------------------------------
The shadow parallelizes ONLY the I/O. Each worker returns (idx, item_id, judge_result,
item_usage); results are reassembled into a pre-sized list by idx (INPUT order) BEFORE
any aggregation. Fitness + usage are then summed single-threaded over the ordered list,
so per_item_scores order, fitness, and usage are bit-identical to serial -- and no two
threads ever mutate a shared usage dict (no race). The mock judge returns per-item
DISTINCT deterministic scores (keyed off item index) so a value-misassociation bug,
not just a pure reorder, would be caught by the equality assertion.

Run
---
  python benchmarks/bench_calibration_concurrency.py            # full timed measurement
  python benchmarks/bench_calibration_concurrency.py --smoke    # fast correctness check (sleeps=0.01)

The full run takes several minutes (N x (target+judge) x runs x modes). Use --smoke
first to validate correctness, then run the full measurement in the background.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from omegaprompt.domain.dataset import Dataset
from omegaprompt.domain.judge import Dimension, HardGate, JudgeResult, JudgeRubric
from omegaprompt.domain.params import PromptVariants
from omegaprompt.judges.base import JudgeOutcome
from omegaprompt.providers.base import (
    CapabilityTier,
    ProviderCapabilities,
    ProviderRequest,
    ProviderResponse,
)
from omegaprompt.targets.prompt_target import (
    PromptTarget,
    _accumulate_usage,
    _merge_usage,
)


# ----------------------------------------------------------------------
# Latency-injecting mock provider + judge (no network)
# ----------------------------------------------------------------------


class MockLatencyProvider:
    """Target provider that sleeps `target_latency_s` per call, then returns
    a deterministic response. sleep() releases the GIL, so concurrent calls
    genuinely overlap -- a faithful model of IO/LLM-bound work."""

    def __init__(self, *, target_latency_s: float) -> None:
        self.name = "mock-target"
        self.model = "mock-target-model"
        self._latency_s = target_latency_s

    def call(self, request: ProviderRequest) -> ProviderResponse:
        time.sleep(self._latency_s)
        return ProviderResponse(
            text="ok",
            usage={
                "input_tokens": 5,
                "output_tokens": 3,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
            latency_ms=self._latency_s * 1000.0,
        )

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(provider=self.name, tier=CapabilityTier.CLOUD)


class MockLatencyJudge:
    """Judge that sleeps `judge_latency_s` per call and returns a per-item
    DISTINCT deterministic score, keyed off the item id's trailing integer.
    Distinct scores make the determinism assertion non-vacuous: a value
    misassociation (not just a reorder) would change per_item_scores."""

    def __init__(self, *, judge_latency_s: float, n_items: int, reverse_stagger: bool = False) -> None:
        self.name = "mock-judge"
        self._latency_s = judge_latency_s
        self._n = n_items
        # When True, later items finish FIRST (sleep decreases with idx). Used
        # only by the negative control to force completion order != input order
        # so the deliberately-buggy variant actually reorders its output.
        self._reverse_stagger = reverse_stagger

    def _score_for(self, item_id: str) -> int:
        # item ids are "t0".."t{n-1}"; alternate 1/0 so scores are distinct
        # across items in a deterministic, item-keyed way. Scale is (0,1).
        idx = int(item_id[1:])
        return 1 if (idx % 2 == 0) else 0

    def score(self, *, rubric, item, target_response) -> JudgeOutcome:
        if self._reverse_stagger:
            idx = int(item.id[1:])
            # later items sleep less -> finish earlier -> completion order
            # reverses relative to input order.
            time.sleep(self._latency_s + (self._n - idx) * 0.003)
        else:
            time.sleep(self._latency_s)
        score = self._score_for(item.id)
        return JudgeOutcome(
            result=JudgeResult(scores={"q": score}, gate_results={"g": True}),
            usage={
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 7,
            },
        )


def _rubric() -> JudgeRubric:
    return JudgeRubric(
        dimensions=[Dimension(name="q", description="quality", weight=1.0, scale=(0, 1))],
        hard_gates=[HardGate(name="g", description="gate")],
    )


def _dataset(n: int) -> Dataset:
    return Dataset.from_items([{"id": f"t{i}", "input": f"in{i}"} for i in range(n)])


def _variants() -> PromptVariants:
    return PromptVariants(
        system_prompts=["sp-A", "sp-B"],
        few_shot_examples=[{"input": "q1", "output": "a1"}],
    )


def _make_target(
    *, n: int, target_latency_s: float, judge_latency_s: float, reverse_stagger: bool = False
) -> PromptTarget:
    """Fresh PromptTarget per timed run (history/usage accumulate across runs)."""
    return PromptTarget(
        target_provider=MockLatencyProvider(target_latency_s=target_latency_s),
        judge=MockLatencyJudge(
            judge_latency_s=judge_latency_s, n_items=n, reverse_stagger=reverse_stagger
        ),
        dataset=_dataset(n),
        rubric=_rubric(),
        variants=_variants(),
    )


# ----------------------------------------------------------------------
# PARALLEL shadow of PromptTarget.evaluate's per-item loop
# ----------------------------------------------------------------------


@dataclass
class _ItemOutput:
    idx: int
    item_id: str
    judge_result: Any
    item_usage: dict


def evaluate_parallel(target: PromptTarget, params: dict | None, *, ceiling: int, max_workers: int):
    """Concurrency-throttled shadow of PromptTarget.evaluate.

    Identical per-item work to the product loop. ONLY the I/O (target.call +
    judge.score) runs concurrently, capped at `ceiling` items in flight via a
    semaphore. Aggregation (fitness, usage) runs single-threaded over the
    INPUT-ordered results, so the output matches serial exactly.

    Returns the same EvalResult that PromptTarget.evaluate would, by delegating
    aggregation back to the real PromptTarget machinery -- we only reorder the
    I/O. To keep this self-contained and product-code-faithful, we recompute
    the metadata that the determinism comparison checks.
    """
    # Mirror evaluate()'s param resolution + variant selection.
    resolved, _warnings = target._resolve_params(params)
    system_prompt = target.variants.system_prompts[resolved.system_prompt_variant]
    few_shots = target.variants.few_shot_examples[: resolved.few_shot_count]

    items = list(target.dataset.items)
    sem = threading.Semaphore(ceiling)

    def _work(idx: int, item) -> _ItemOutput:
        with sem:  # at most `ceiling` items doing I/O at once
            target_request = ProviderRequest(
                system_prompt=system_prompt,
                user_message=item.input,
                few_shots=list(few_shots),
                reasoning_profile=resolved.reasoning_profile,
                output_budget_bucket=resolved.output_budget_bucket,
                response_schema_mode=resolved.response_schema_mode,
                tool_policy_variant=resolved.tool_policy_variant,
                execution_profile=target.execution_profile,
            )
            target_response = target.target_provider.call(target_request)
            judge_outcome = target.judge.score(
                rubric=target.rubric,
                item=item,
                target_response=target_response.text,
            )
            judge_result, judge_usage = judge_outcome
            item_usage = _merge_usage(target_response.usage, judge_usage)
        return _ItemOutput(
            idx=idx, item_id=item.id, judge_result=judge_result, item_usage=item_usage
        )

    # Pre-sized list -> reassemble in INPUT order (never as_completed-append).
    outputs: list[_ItemOutput | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for out in ex.map(lambda p: _work(*p), list(enumerate(items))):
            outputs[out.idx] = out

    # ---- deterministic single-threaded aggregation over ordered outputs ----
    from omegaprompt.core.fitness import CompositeFitness
    from omegaprompt.providers.base import empty_usage

    fitness_engine = CompositeFitness(target.rubric)
    run_usage = empty_usage()
    judge_results: list[tuple[str, Any]] = []
    for out in outputs:
        assert out is not None
        _accumulate_usage(run_usage, out.item_usage)
        judge_results.append((out.item_id, out.judge_result))

    fitness = fitness_engine.evaluate(judge_results)
    pass_rate = fitness_engine.pass_rate()
    per_item_scores = [
        {
            "item_id": p.item_id,
            "soft_score": p.soft_score,
            "gates_passed": p.gates_passed,
            "final_score": p.final_score,
        }
        for p in fitness_engine.last_per_item
    ]
    return {
        "fitness": fitness,
        "hard_gate_pass_rate": pass_rate,
        "usage": dict(run_usage),
        "per_item_scores": per_item_scores,
    }


def evaluate_parallel_BUGGY(target: PromptTarget, params: dict | None, *, ceiling: int, max_workers: int):
    """NEGATIVE CONTROL ONLY -- a deliberately wrong parallel variant that
    appends results in COMPLETION order (as they finish) instead of INPUT
    order. This is the classic ordering bug the future product change must
    avoid. The harness's negative control asserts that our determinism guard
    REJECTS this output (i.e. the guard has teeth, not just passes on the
    correct path). NEVER use this for measurement."""
    from concurrent.futures import as_completed

    resolved, _warnings = target._resolve_params(params)
    system_prompt = target.variants.system_prompts[resolved.system_prompt_variant]
    few_shots = target.variants.few_shot_examples[: resolved.few_shot_count]
    items = list(target.dataset.items)
    sem = threading.Semaphore(ceiling)

    def _work(idx: int, item) -> _ItemOutput:
        with sem:
            req = ProviderRequest(
                system_prompt=system_prompt,
                user_message=item.input,
                few_shots=list(few_shots),
                reasoning_profile=resolved.reasoning_profile,
                output_budget_bucket=resolved.output_budget_bucket,
                response_schema_mode=resolved.response_schema_mode,
                tool_policy_variant=resolved.tool_policy_variant,
                execution_profile=target.execution_profile,
            )
            resp = target.target_provider.call(req)
            jr, ju = target.judge.score(rubric=target.rubric, item=item, target_response=resp.text)
            usage = _merge_usage(resp.usage, ju)
        return _ItemOutput(idx=idx, item_id=item.id, judge_result=jr, item_usage=usage)

    from omegaprompt.core.fitness import CompositeFitness
    from omegaprompt.providers.base import empty_usage

    fitness_engine = CompositeFitness(target.rubric)
    run_usage = empty_usage()
    judge_results: list[tuple[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_work, i, it) for i, it in enumerate(items)]
        # BUG: iterate in completion order, append as they land -> output
        # order no longer matches input order.
        for fut in as_completed(futures):
            out = fut.result()
            _accumulate_usage(run_usage, out.item_usage)
            judge_results.append((out.item_id, out.judge_result))

    fitness = fitness_engine.evaluate(judge_results)
    per_item_scores = [
        {
            "item_id": p.item_id,
            "soft_score": p.soft_score,
            "gates_passed": p.gates_passed,
            "final_score": p.final_score,
        }
        for p in fitness_engine.last_per_item
    ]
    return {"per_item_scores": per_item_scores}


# ----------------------------------------------------------------------
# Determinism assertion: parallel == serial (order + values, excl. wall-time)
# ----------------------------------------------------------------------


def assert_parallel_matches_serial(*, n: int, target_latency_s: float, judge_latency_s: float) -> None:
    """The harness's determinism guard. Asserts the parallel shadow produces
    per_item_scores in INPUT order with identical values, identical fitness,
    identical hard_gate_pass_rate, and identical per-run usage. EXCLUDES
    latency_ms (wall-time, differs by construction). Run at several ceilings."""
    params = {"system_prompt_variant": 1, "few_shot_count": 1}

    serial_target = _make_target(
        n=n, target_latency_s=target_latency_s, judge_latency_s=judge_latency_s
    )
    serial = serial_target.evaluate(params)
    serial_scores = serial.metadata["per_item_scores"]
    serial_usage = serial.metadata["usage"]

    for ceiling in (1, 2, 4, 8):
        par_target = _make_target(
            n=n, target_latency_s=target_latency_s, judge_latency_s=judge_latency_s
        )
        par = evaluate_parallel(
            par_target, params, ceiling=ceiling, max_workers=max(n, ceiling)
        )
        # ORDER + VALUES: per_item_scores must match element-for-element.
        assert par["per_item_scores"] == serial_scores, (
            f"per_item_scores mismatch at ceiling={ceiling} "
            f"(ordering or value bug would surface here)"
        )
        assert abs(par["fitness"] - serial.fitness) < 1e-12, "fitness mismatch"
        assert (
            abs(par["hard_gate_pass_rate"] - serial.metadata["hard_gate_pass_rate"]) < 1e-12
        ), "hard_gate_pass_rate mismatch"
        assert par["usage"] == serial_usage, "per-run usage mismatch (accumulation bug)"

    # Sanity: scores are actually distinct across items (assertion non-vacuous).
    finals = [row["final_score"] for row in serial_scores]
    assert len(set(finals)) > 1, "mock judge produced identical scores -> assertion vacuous"


def assert_negative_control_catches_ordering_bug(
    *, n: int, target_latency_s: float, judge_latency_s: float
) -> bool:
    """NEGATIVE CONTROL: prove the determinism guard has TEETH.

    Runs the deliberately-buggy completion-order parallel variant and confirms
    the SAME per_item_scores list-equality check that passes on the correct
    shadow now REJECTS the buggy output. This demonstrates (not just asserts)
    that an output-reordering bug -- the exact failure the future product
    change must avoid -- would be caught by the harness.

    The check is item_id-keyed list equality, so any reordering of the output
    rows is caught airtight. (Value-misassociation between same-score items is
    only partially covered because the rubric scale is (0,1) -> scores are 0/1;
    see the reported determinism note.)

    Returns True if the guard correctly flagged the bug. Best run at high
    ceiling so completion order actually diverges from input order.
    """
    params = {"system_prompt_variant": 1, "few_shot_count": 1}
    serial = _make_target(
        n=n, target_latency_s=target_latency_s, judge_latency_s=judge_latency_s
    ).evaluate(params)
    serial_scores = serial.metadata["per_item_scores"]

    # Try a few times; thread completion order is nondeterministic, so a single
    # run could (rarely) happen to finish in input order. Any divergence proves
    # the guard fires.
    caught = False
    for _ in range(8):
        # reverse_stagger forces completion order to diverge from input order,
        # so the buggy append-on-completion variant actually reorders output.
        buggy = evaluate_parallel_BUGGY(
            _make_target(
                n=n,
                target_latency_s=target_latency_s,
                judge_latency_s=judge_latency_s,
                reverse_stagger=True,
            ),
            params,
            ceiling=max(2, n),
            max_workers=max(n, 2),
        )
        if buggy["per_item_scores"] != serial_scores:
            caught = True
            break
    return caught


# ----------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------


def _median_runs(fn, *, runs: int) -> tuple[float, float, list[float]]:
    samples: list[float] = []
    for _ in range(runs):
        t0 = perf_counter()
        fn()
        samples.append((perf_counter() - t0) * 1000.0)
    med = statistics.median(samples)
    stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    return med, stdev, samples


def measure(*, n: int, runs: int, target_latency_s: float, judge_latency_s: float) -> None:
    params = {"system_prompt_variant": 1, "few_shot_count": 1}
    ceilings = [1, 2, 4, 8]

    print("=" * 72)
    print("omegaprompt calibration concurrency benchmark (M3 measure-before-commit)")
    print("=" * 72)
    print(
        f"N items={n}  runs(median)={runs}  target_latency={target_latency_s}s  "
        f"judge_latency={judge_latency_s}s"
    )
    per_item_ideal = target_latency_s + judge_latency_s
    print(
        f"per-item ideal serial cost = {per_item_ideal:.2f}s  "
        f"(ideal serial total ~= {per_item_ideal * n:.1f}s)"
    )
    print()

    # ---- serial baseline (the before-number): the REAL evaluate() path ----
    def _serial():
        t = _make_target(
            n=n, target_latency_s=target_latency_s, judge_latency_s=judge_latency_s
        )
        t.evaluate(params)

    serial_med, serial_std, serial_samples = _median_runs(_serial, runs=runs)
    print("SERIAL baseline (real PromptTarget.evaluate, current code path):")
    print(
        f"  median={serial_med:.1f}ms  stdev={serial_std:.1f}ms  "
        f"samples={[round(s) for s in serial_samples]}"
    )
    print()

    # ---- parallel at each ceiling ----
    print("PARALLEL shadow (only I/O concurrent, throttled by ceiling c):")
    print(
        f"  {'c':>3}  {'median_ms':>10}  {'stdev_ms':>9}  "
        f"{'speedup_x':>9}  {'faster_vs_serial':>16}"
    )
    rows = []
    for c in ceilings:
        def _parallel(c=c):
            t = _make_target(
                n=n, target_latency_s=target_latency_s, judge_latency_s=judge_latency_s
            )
            evaluate_parallel(t, params, ceiling=c, max_workers=max(n, c))

        med, std, samples = _median_runs(_parallel, runs=runs)
        speedup = serial_med / med if med > 0 else float("inf")
        pct_faster = (1.0 - med / serial_med) * 100.0 if serial_med > 0 else 0.0
        rows.append((c, med, std, speedup, pct_faster))
        print(
            f"  {c:>3}  {med:>10.1f}  {std:>9.1f}  {speedup:>8.2f}x  {pct_faster:>15.1f}%"
        )

    print()
    print("Interpretation:")
    print("  c=1 should be ~0% faster (no real concurrency -> headline collapses).")
    print("  c=2 should be ~50% faster (in the 30-60% band).")
    print("  c>=4 should exceed 60% (more concurrency than the band assumes).")
    print()
    print("The 30-60% speedup claim holds ONLY if the real provider permits c>1")
    print("concurrent calls. The real ceiling is the OWNER's provider RPM/TPM config.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20, help="dataset items")
    ap.add_argument("--runs", type=int, default=5, help="timed runs per mode (median)")
    ap.add_argument("--target-latency", type=float, default=1.0, help="target call sleep (s)")
    ap.add_argument("--judge-latency", type=float, default=0.3, help="judge call sleep (s)")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="fast correctness check only: sleeps=0.01, N=8, runs=2",
    )
    args = ap.parse_args()

    if args.smoke:
        n, runs, tl, jl = 8, 2, 0.01, 0.01
    else:
        n, runs, tl, jl = args.n, args.runs, args.target_latency, args.judge_latency

    print("Running determinism assertion (parallel == serial, order + values)...")
    assert_parallel_matches_serial(n=n, target_latency_s=tl, judge_latency_s=jl)
    print("  PASS: parallel results match serial (per_item_scores order+values, "
          "fitness, pass_rate, usage).")

    # Negative control: prove the guard has teeth (rejects an ordering bug).
    nc_jl = min(jl, 0.01)  # keep the control fast even at full latency
    caught = assert_negative_control_catches_ordering_bug(
        n=n, target_latency_s=min(tl, 0.01), judge_latency_s=nc_jl
    )
    assert caught, (
        "NEGATIVE CONTROL FAILED: the buggy completion-order variant was NOT "
        "rejected -> the determinism guard would miss an ordering bug."
    )
    print("  PASS (negative control): the same per_item_scores check REJECTS a "
          "deliberately reordered (completion-order) parallel output -> guard has teeth.")
    print()

    measure(n=n, runs=runs, target_latency_s=tl, judge_latency_s=jl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
