"""Reviewer follow-up to PR5: ``requires_manual_review`` must be
honoured at the artifact layer, not just on the in-process plan object.

PR5 added the field on AdaptationPlan but no consumer read it. PR7
wires ``apply_ship_gate_escalation`` into ``runtime.calibrate`` so
the artifact's ``status`` and ``ship_recommendation`` reflect the
plan's manual-review reasons. CI gates keying off the artifact now
behave the same way the in-process ``plan.requires_manual_review``
property would.
"""

from __future__ import annotations

import pytest

from omegaprompt.domain.profiles import ShipRecommendation
from omegaprompt.preflight import (
    AdaptationPlan,
    ParameterOverride,
    apply_ship_gate_escalation,
)


# ---------------------------------------------------------------------------
# Pure helper: apply_ship_gate_escalation.
# ---------------------------------------------------------------------------


def _plan_no_review() -> AdaptationPlan:
    return AdaptationPlan(
        require_manual_review_reasons=[],
        overrides=[],
        rationale=[],
    )


def _plan_review_required() -> AdaptationPlan:
    return AdaptationPlan(
        require_manual_review_reasons=[
            "small_sample_kc4_power: held-out slice too small for KC-4 power",
        ],
        overrides=[
            ParameterOverride(
                parameter="manual_review",
                default=False,
                applied=True,
                reason="small sample",
            )
        ],
        rationale=["max_gap NOT widened on small sample"],
    )


def test_apply_ship_gate_escalation_passes_through_when_review_not_required():
    plan = _plan_no_review()
    status, ship, rationale, extras = apply_ship_gate_escalation(
        plan,
        status="OK",
        ship_recommendation=ShipRecommendation.SHIP,
        rationale="passed",
    )
    assert status == "OK"
    assert ship == ShipRecommendation.SHIP
    assert rationale == "passed"
    assert extras == []


def test_apply_ship_gate_escalation_forces_hold_and_status_when_review_required():
    plan = _plan_review_required()
    status, ship, rationale, extras = apply_ship_gate_escalation(
        plan,
        status="OK",
        ship_recommendation=ShipRecommendation.SHIP,
        rationale="passed",
    )
    assert status == "REQUIRES_MANUAL_REVIEW"
    assert ship == ShipRecommendation.HOLD
    assert "manual review required" in rationale
    assert "small_sample_kc4_power" in rationale
    assert len(extras) == 1


def test_apply_ship_gate_escalation_preserves_failed_status():
    """A FAIL_KC4_GATE candidate stays at that status — escalation can
    only sharpen, not soften, the existing verdict."""
    plan = _plan_review_required()
    status, ship, _r, _e = apply_ship_gate_escalation(
        plan,
        status="FAIL_KC4_GATE",
        ship_recommendation=ShipRecommendation.HOLD,
        rationale="kc4 below threshold",
    )
    assert status == "FAIL_KC4_GATE"
    assert ship == ShipRecommendation.HOLD


# ---------------------------------------------------------------------------
# Integration: runtime.calibrate honours the plan when supplied.
# ---------------------------------------------------------------------------


def test_calibrate_signature_accepts_adaptation_plan_kwarg():
    """Pin the public surface — agents must be able to pass a plan."""
    import inspect

    from omegaprompt.runtime import calibrate

    params = inspect.signature(calibrate).parameters
    assert "adaptation_plan" in params
    # default must be None so existing call sites are backward-compat:
    assert params["adaptation_plan"].default is None


def test_apply_ship_gate_escalation_collects_all_review_reasons():
    plan = AdaptationPlan(
        require_manual_review_reasons=[
            "small_sample_kc4_power: too few items",
            "future_trap: another reason",
        ],
        overrides=[],
        rationale=[],
    )
    _s, _ship, rationale, extras = apply_ship_gate_escalation(
        plan,
        status="OK",
        ship_recommendation=ShipRecommendation.SHIP,
        rationale="passed",
    )
    assert len(extras) == 2
    assert "small_sample_kc4_power" in rationale
    assert "future_trap" in rationale


@pytest.mark.parametrize(
    "input_ship",
    [
        ShipRecommendation.SHIP,
        ShipRecommendation.EXPERIMENT,
        ShipRecommendation.BLOCK,
    ],
)
def test_apply_ship_gate_escalation_always_lands_on_hold_when_review_required(input_ship):
    plan = _plan_review_required()
    _s, ship, _r, _e = apply_ship_gate_escalation(
        plan,
        status="OK",
        ship_recommendation=input_ship,
        rationale="r",
    )
    assert ship == ShipRecommendation.HOLD
