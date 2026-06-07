"""Fail-loud contract test for the omega-lock dependency seam.

omegaprompt implements omega-lock's ``CalibrableTarget`` protocol and consumes
specific fields off omega-lock's result types. Those fields are omega-lock's
*internal* surface, not a stable public API, and ``.venv`` is gitignored so
ripgrep / static analysis cannot see them. This test is the only mechanism that
watches the seam: it fails LOUD (named, located, pre-merge) when a future
omega-lock upgrade drifts a field omegaprompt depends on, instead of letting it
surface as a silent mid-run ``AttributeError``.

History: omega-lock 0.1.4 -> 0.3.0 renamed the target-result action count
``n_trials`` -> ``sample_count`` and broke omegaprompt's calibration path
(``omega_lock/stress.py`` read ``.sample_count`` off a result that only had
``.n_trials``). The first round-trip assertion below would have caught that
before merge.

Each assertion names the omegaprompt consumer it protects. Zero network.
"""

from __future__ import annotations

import dataclasses
import inspect

import omega_lock
from omega_lock import EvalResult as OmegaLockEvalResult
from omega_lock import (
    CalibrableTarget,
    KCThresholds,
    P1Config,
    P1Result,
    ParamSpec,
    check_kc4,
    measure_stress,
    run_p1,
)
from omega_lock.walk_forward import WalkForwardResult as OmegaLockWalkForwardResult

from omegaprompt.domain.result import EvalResult as OmegaPromptEvalResult


def _field_names(cls) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def test_omegaprompt_evalresult_exposes_sample_count_alias() -> None:
    """LOAD-BEARING. omega-lock >=0.3.0 reads ``.sample_count`` off the result
    ``PromptTarget.evaluate`` returns (``src/omegaprompt/targets/prompt_target.py``
    sets ``n_trials=len(dataset)``). omegaprompt stores ``n_trials`` and aliases
    ``sample_count`` (``src/omegaprompt/domain/result.py``). If this round-trip
    breaks, calibration crashes in ``omega_lock/stress.py``. Asserting only
    omega-lock's own field would NOT have caught the 0.3.0 break.
    """
    r = OmegaPromptEvalResult(params={}, item_results=[], fitness=1.0, n_trials=7)
    assert r.sample_count == 7 == r.n_trials
    # omega-lock reads exactly these two attributes off a target's result:
    assert hasattr(r, "fitness")
    assert hasattr(r, "sample_count")


def test_omega_lock_evalresult_exposes_sample_count() -> None:
    """omega-lock's own ``EvalResult`` must expose ``sample_count`` — the canonical
    field its stress/grid/walk_forward consumers read. Construct with the canonical
    name (NOT the deprecated ``n_trials`` alias) so a clean alias removal in a
    future omega-lock does not false-fail this guard. omegaprompt's OWN result keeps
    the ``n_trials`` field; that round-trip is asserted separately above."""
    assert "sample_count" in _field_names(OmegaLockEvalResult)
    r = OmegaLockEvalResult(fitness=1.0, sample_count=5)
    assert r.sample_count == 5


def test_walkforwardresult_keys_provenance_forward_tripwire() -> None:
    """FORWARD TRIPWIRE (not yet a live guard). A future provenance-recording
    path WILL record these ``WalkForwardResult`` keys via
    ``P1Result.walk_forward = wf.to_dict()``; omegaprompt does not consume
    omega-lock's ``WalkForwardResult`` yet (grep src: zero hits — every current
    ``.walk_forward`` access is on omegaprompt's OWN per-item type).
    ``pearson_computable``/``pearson_status``/``test_best_*`` are 0.3.0 additions
    that path will consume rather than rebuild — assert now so it cannot land
    against a drifted surface."""
    expected = {
        "train_fitnesses",
        "test_fitnesses",
        "pearson",
        "pearson_status",
        "pearson_computable",
        "test_best_fitness",
        "test_best_params",
        "trade_ratio_scaled",
    }
    # Assert OUTPUT keys (wf.to_dict() — what the provenance path actually records),
    # not dataclasses.fields(): survives a perf-motivated representation swap that
    # preserves to_dict() output (friction-review item 2, third flip).
    wf = OmegaLockWalkForwardResult(
        top_n=1,
        train_fitnesses=[1.0],
        test_fitnesses=[1.0],
        test_n_trials=[1],
        pearson=0.0,
        pearson_status="OK",
        pearson_computable=True,
        train_best_trades_mean=1.0,
        test_best_trades=1,
        test_best_fitness=1.0,
        test_best_params={},
        trade_ratio_scaled=1.0,
    )
    missing = expected - set(wf.to_dict())
    assert not missing, f"omega-lock WalkForwardResult.to_dict() dropped consumed keys: {missing}"


def test_p1result_currently_consumed_fields() -> None:
    """LIVE GUARD. omegaprompt reads these off ``P1Result`` TODAY: ``grid_best``
    (runtime.py:325) and ``stress_results`` (runtime.py:359 -> _sensitivity_rows
    -> sensitivity_ranking). A rename of either is the exact silent-drift class
    (n_trials -> sample_count) this file guards: a ``stress_results`` rename would
    silently empty ``sensitivity_ranking``."""
    expected = {"grid_best", "stress_results"}
    missing = expected - _field_names(P1Result)
    assert not missing, f"omega-lock P1Result dropped a CURRENTLY-consumed field: {missing}"


def test_grid_best_unlocked_wire_key_consumed() -> None:
    """LIVE GUARD (nested wire-key — the silent class). omegaprompt reads
    ``grid_best.get("unlocked")`` off ``P1Result.grid_best`` (runtime.py:327),
    which is ``GridPoint.to_summary()`` output (omega_lock/orchestrator.py:633).
    ``"unlocked"`` is a MANUAL literal in ``to_summary`` (omega_lock/grid.py:38),
    so ``_field_names``/``dataclasses.fields`` is structurally blind to it: a
    rename would silently empty ``best_unlocked`` and fall the calibrated winner
    back to neutral with NO error and a valid-looking artifact. Assert the OUTPUT
    key directly (build a real GridPoint; never reflect on fields)."""
    from omega_lock.grid import GridPoint

    summary = GridPoint(
        idx=0,
        unlocked={"a": 1},
        params={"a": 1},
        result=OmegaLockEvalResult(fitness=1.0, sample_count=3),
    ).to_summary()
    assert "unlocked" in summary, (
        "omega-lock GridPoint.to_summary() dropped the 'unlocked' wire-key that "
        "omegaprompt reads at runtime.py:327 — the calibrated winner would silently "
        "fall back to neutral."
    )


def test_stressresult_surface_consumed_by_sensitivity() -> None:
    """LIVE GUARD. The sensitivity path reads ``name`` / ``normalized_stress`` /
    ``raw_stress`` off omega-lock ``StressResult`` (runtime.py:752,758-760, and the
    dict form via ``P1Result.stress_results`` -> _sensitivity_rows). A rename
    silently empties or mis-ranks ``sensitivity_ranking``."""
    from omega_lock.stress import StressResult

    missing = {"name", "normalized_stress", "raw_stress"} - _field_names(StressResult)
    assert not missing, f"omega-lock StressResult dropped a consumed field: {missing}"
    # Also assert the OUTPUT wire form the consumer actually reads
    # (P1Result.stress_results = [s.to_dict() ...] -> _sensitivity_rows reads the
    # dict keys). A representation swap that kept the dataclass-field check green
    # could still drop a to_dict() key.
    wire = StressResult(
        name="x",
        baseline_fitness=1.0,
        plus_fitness=1.0,
        minus_fitness=1.0,
        epsilon=0.1,
        raw_stress=0.0,
    ).to_dict()
    wire_missing = {"name", "normalized_stress", "raw_stress"} - set(wire)
    assert not wire_missing, f"omega-lock StressResult.to_dict() dropped a consumed key: {wire_missing}"


def test_p1result_provenance_fields_forward_tripwire() -> None:
    """FORWARD TRIPWIRE (not yet a live guard). A future provenance-recording path
    WILL record these off ``P1Result`` once wired; omegaprompt does not consume
    them yet (grep src: zero hits). Asserting now so it cannot land against a
    surface that already drifted."""
    expected = {"walk_forward", "kc_reports", "omega_lock_version", "holdout_result"}
    missing = expected - _field_names(P1Result)
    assert not missing, f"omega-lock P1Result missing provenance fields: {missing}"


def test_check_kc4_signature_subset() -> None:
    """A future provenance path reasons about ``check_kc4``'s inputs: it would
    record the per-CANDIDATE pearson as provenance (never equated with omegaprompt's
    own per-item gate). Assert SUBSET, not ``==``: omegaprompt does not CALL
    check_kc4, so a backward-compatible parameter ADD must not false-fail this guard."""
    params = set(inspect.signature(check_kc4).parameters)
    assert {"train_fitnesses", "test_fitnesses", "trade_ratio", "thresholds"} <= params


def test_kcthresholds_consumed_surface() -> None:
    """omegaprompt's ``pure_objective()`` preset and the trade_ratio analysis
    depend on these knobs. ``pure_objective`` must SKIP the action-count gates
    (KC-3 ``trade_count_min`` / KC-4b ``trade_ratio_min`` -> None) so a small
    prompt dataset is not nuked by trade-count floors."""
    fields = _field_names(KCThresholds)
    assert {"min_nonzero_stress_count", "trade_ratio_min", "pearson_min", "trade_count_min"} <= fields
    assert hasattr(KCThresholds, "pure_objective")
    pure = KCThresholds.pure_objective()
    assert pure.trade_ratio_min is None
    assert pure.trade_count_min is None
    base = KCThresholds()
    assert base.trade_ratio_min == 0.5
    assert base.pearson_min == 0.3


def test_consumed_callable_signatures_subset() -> None:
    """A3 net-new (a). omegaprompt drives these omega-lock callables; assert the
    consumed parameters are PRESENT (subset, not ``==``, so a backward-compatible
    parameter add does not false-fail). Mirrors the producer-side manifest (A1/A2)."""

    def _params(fn) -> set[str]:
        return set(inspect.signature(fn).parameters)

    assert {"train_target", "config", "test_target"} <= _params(run_p1)
    assert {"unlock_k"} <= _params(P1Config)
    assert {"target", "baseline_params", "baseline_result"} <= _params(measure_stress)
    assert {"name", "dtype", "neutral", "low", "high"} <= _params(ParamSpec)


def test_calibrabletarget_protocol_method_presence() -> None:
    """A3 net-new (b). omegaprompt's PromptTarget must satisfy omega-lock's
    ``@runtime_checkable`` ``CalibrableTarget`` protocol. The check is method-name
    presence only (``param_space`` / ``evaluate``); a producer-side protocol method
    rename would break the seam and is caught here."""

    class _Stub:
        def param_space(self):
            return []

        def evaluate(self, params):
            return OmegaLockEvalResult(fitness=1.0, sample_count=1)

    assert isinstance(_Stub(), CalibrableTarget)


def test_installed_omega_lock_within_verified_contract_range() -> None:
    """The installed version must match the contract-verified pin range
    (``omega-lock>=0.3.0,<0.4.0`` in pyproject.toml). A 0.4.x install means this
    seam was never re-verified — fail loud and force re-verification."""
    assert omega_lock.__version__.startswith("0.3."), (
        f"omega-lock {omega_lock.__version__} is outside the contract-verified range "
        "(>=0.3.0,<0.4.0). Re-verify the seam assertions in this file, then widen the "
        "pin in pyproject.toml."
    )
