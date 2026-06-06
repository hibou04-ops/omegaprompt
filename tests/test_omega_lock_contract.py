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
from omega_lock import KCThresholds, P1Result, check_kc4
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


def test_omega_lock_evalresult_has_sample_count_and_n_trials_alias() -> None:
    """omega-lock's own ``EvalResult`` must expose ``sample_count`` (the field
    its stress/grid/walk_forward consumers read) and accept ``n_trials`` as the
    deprecated alias omegaprompt still constructs with."""
    assert "sample_count" in _field_names(OmegaLockEvalResult)
    r = OmegaLockEvalResult(fitness=1.0, n_trials=5)
    assert r.sample_count == 5


def test_walkforwardresult_exposes_keys_adoption_spine_records() -> None:
    """The adoption spine records these ``WalkForwardResult`` keys onto the
    artifact as provenance (consumed via ``P1Result.walk_forward = wf.to_dict()``).
    ``pearson_computable``/``pearson_status`` and ``test_best_*`` are 0.3.0
    additions the plan consumes rather than rebuilds — drift silently drops them."""
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
    missing = expected - _field_names(OmegaLockWalkForwardResult)
    assert not missing, f"omega-lock WalkForwardResult dropped consumed fields: {missing}"


def test_p1result_exposes_fields_runtime_reads() -> None:
    """omegaprompt reads these off ``P1Result``: ``grid_best`` (runtime calibrate),
    ``walk_forward`` / ``kc_reports`` / ``omega_lock_version`` (adoption spine +
    version stamp), ``holdout_result`` (deferred-conditional). Drift breaks
    calibrate or silently drops provenance."""
    expected = {"grid_best", "walk_forward", "holdout_result", "kc_reports", "omega_lock_version"}
    missing = expected - _field_names(P1Result)
    assert not missing, f"omega-lock P1Result dropped consumed fields: {missing}"


def test_check_kc4_signature_stable() -> None:
    """The adoption spine reasons about ``check_kc4``'s inputs: it records the
    per-CANDIDATE pearson as provenance (never equated with omegaprompt's own
    per-item gate). A signature change means the recorded statistic changed."""
    assert list(inspect.signature(check_kc4).parameters) == [
        "train_fitnesses",
        "test_fitnesses",
        "trade_ratio",
        "thresholds",
    ]


def test_kcthresholds_consumed_surface() -> None:
    """omegaprompt's ``pure_objective()`` adoption and the trade_ratio analysis
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


def test_installed_omega_lock_within_verified_contract_range() -> None:
    """The installed version must match the contract-verified pin range
    (``omega-lock>=0.3.0,<0.4.0`` in pyproject.toml). A 0.4.x install means this
    seam was never re-verified — fail loud and force re-verification."""
    assert omega_lock.__version__.startswith("0.3."), (
        f"omega-lock {omega_lock.__version__} is outside the contract-verified range "
        "(>=0.3.0,<0.4.0). Re-verify the seam assertions in this file, then widen the "
        "pin in pyproject.toml."
    )
