"""Reviewer P2: `omegaprompt diff` must treat artifact status and
ship_recommendation as first-class regression conditions.

Without these checks, a candidate with status=FAIL_KC4_GATE but better raw
fitness slips through CI as 'no regression' — a false-safe outcome that
defeats the point of a CI gate.
"""

from __future__ import annotations

from omegaprompt.domain import ShipRecommendation
from omegaprompt.domain.result import CalibrationArtifact
from omegaprompt.runtime import diff


def _make_artifact(
    *,
    status: str = "OK",
    ship: ShipRecommendation = ShipRecommendation.SHIP,
    calibrated_fitness: float = 0.8,
    hard_gate_pass_rate: float = 1.0,
    quality_per_cost: float = 0.2,
    quality_per_latency: float = 0.1,
) -> CalibrationArtifact:
    return CalibrationArtifact(
        method="p1",
        unlock_k=3,
        best_params={"system_prompt_variant": 1},
        best_fitness=calibrated_fitness,
        calibrated_fitness=calibrated_fitness,
        neutral_fitness=0.5,
        hard_gate_pass_rate=hard_gate_pass_rate,
        quality_per_cost_best=quality_per_cost,
        quality_per_latency_best=quality_per_latency,
        n_candidates_evaluated=1,
        total_api_calls=1,
        status=status,
        ship_recommendation=ship,
    )


def test_diff_flags_regression_when_candidate_status_is_not_ok():
    """Candidate with FAIL_KC4_GATE must regress even if metrics improve."""
    old = _make_artifact(status="OK", calibrated_fitness=0.7)
    new = _make_artifact(status="FAIL_KC4_GATE", calibrated_fitness=0.95)
    result = diff(old, new)
    assert result.regressed
    assert any("status is not OK" in r for r in result.regression_reasons)
    assert any("FAIL_KC4_GATE" in r for r in result.regression_reasons)


def test_diff_flags_regression_when_candidate_ship_recommendation_is_block():
    old = _make_artifact(ship=ShipRecommendation.SHIP, calibrated_fitness=0.7)
    new = _make_artifact(ship=ShipRecommendation.BLOCK, calibrated_fitness=0.95)
    result = diff(old, new)
    assert result.regressed
    assert any("ship_recommendation=block" in r for r in result.regression_reasons)


def test_diff_flags_regression_when_candidate_ship_recommendation_is_hold():
    old = _make_artifact(ship=ShipRecommendation.SHIP, calibrated_fitness=0.7)
    new = _make_artifact(ship=ShipRecommendation.HOLD, calibrated_fitness=0.95)
    result = diff(old, new)
    assert result.regressed
    assert any("ship_recommendation=hold" in r for r in result.regression_reasons)


def test_diff_does_not_flag_status_when_candidate_is_ok_and_ship():
    old = _make_artifact(calibrated_fitness=0.7)
    new = _make_artifact(calibrated_fitness=0.85)
    result = diff(old, new)
    assert not result.regressed
    assert result.regression_reasons == []


def test_diff_experiment_ship_recommendation_does_not_regress():
    """EXPERIMENT is an opt-in non-blocking signal — distinct from BLOCK/HOLD."""
    old = _make_artifact(ship=ShipRecommendation.EXPERIMENT, calibrated_fitness=0.7)
    new = _make_artifact(ship=ShipRecommendation.EXPERIMENT, calibrated_fitness=0.8)
    result = diff(old, new)
    assert not result.regressed


def test_diff_status_failure_overrides_metric_improvement_in_markdown():
    old = _make_artifact(calibrated_fitness=0.5)
    new = _make_artifact(status="FAIL_HARD_GATES", calibrated_fitness=0.99)
    md = diff(old, new, format="markdown")
    assert "REGRESSION" in md
    assert "FAIL_HARD_GATES" in md
