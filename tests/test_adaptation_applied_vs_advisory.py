"""Reviewer P0: AdaptationPlan distinguishes applied vs advisory overrides.

Pre-fix every entry on ``AdaptationPlan.overrides`` looked equally
authoritative, but only ``min_kc4``, ``max_gap``, ``unlock_k`` actually
flowed through ``runtime.calibrate`` via ``apply_adaptation_plan``.
A reviewer reading the artifact had no way to tell which overrides
changed pipeline behaviour and which were recorded for awareness.

Post-fix:
- ``AdaptationPlan.split_overrides()`` partitions into (applied, advisory).
- ``CalibrationArtifact.adaptation_summary`` records both lists +
  manual_review state.
- ``APPLIED_PARAMETERS`` is the source-of-truth set.
"""
from __future__ import annotations

from omegaprompt.preflight import AdaptationPlan, ParameterOverride
from omegaprompt.preflight.adaptation import APPLIED_PARAMETERS


def test_applied_parameters_set_is_documented():
    """The set of applied parameter names must be visible to reviewers
    so they know what overrides actually move the pipeline."""
    assert "min_kc4" in APPLIED_PARAMETERS
    assert "max_gap" in APPLIED_PARAMETERS
    assert "unlock_k" in APPLIED_PARAMETERS


def test_split_overrides_partitions_known_advisory():
    """Overrides recorded by derive_adaptation_plan today fall into
    two camps: applied (consumed by apply_adaptation_plan) and advisory
    (recorded but no consumer)."""
    plan = AdaptationPlan(
        overrides=[
            ParameterOverride(parameter="min_kc4", default=0.5, applied=0.6, reason="x"),
            ParameterOverride(parameter="rescore_count", default=1, applied=3, reason="x"),
            ParameterOverride(parameter="unlock_k", default=3, applied=2, reason="x"),
            ParameterOverride(parameter="schema_mode", default="strict_schema", applied="json_object", reason="x"),
            ParameterOverride(parameter="judge_ensemble_shift", default=0.0, applied=0.4, reason="x"),
        ],
    )
    applied, advisory = plan.split_overrides()
    assert {ov.parameter for ov in applied} == {"min_kc4", "unlock_k"}
    assert {ov.parameter for ov in advisory} == {
        "rescore_count", "schema_mode", "judge_ensemble_shift",
    }


def test_split_overrides_handles_empty():
    plan = AdaptationPlan()
    applied, advisory = plan.split_overrides()
    assert applied == []
    assert advisory == []


def test_split_overrides_manual_review_recognised_as_applied():
    """Manual review escalation flows through apply_ship_gate_escalation
    so it's an applied behaviour change, not advisory-only."""
    plan = AdaptationPlan(
        overrides=[
            ParameterOverride(
                parameter="manual_review",
                default=False, applied=True, reason="small sample",
            ),
        ],
    )
    applied, advisory = plan.split_overrides()
    assert {ov.parameter for ov in applied} == {"manual_review"}
    assert advisory == []


def test_adaptation_summary_helper_renders_split_for_artifact():
    """The runtime helper that builds the artifact's
    ``adaptation_summary`` field returns the right shape."""
    from omegaprompt.runtime import _build_adaptation_summary

    plan = AdaptationPlan(
        overrides=[
            ParameterOverride(parameter="min_kc4", default=0.5, applied=0.6, reason="x"),
            ParameterOverride(parameter="rescore_count", default=1, applied=3, reason="x"),
        ],
        require_manual_review_reasons=["small sample held-out"],
    )
    summary = _build_adaptation_summary(plan)
    assert summary is not None
    assert summary["applied"] == ["min_kc4"]
    assert summary["advisory_not_applied"] == ["rescore_count"]
    assert summary["manual_review_required"] is True
    assert summary["manual_review_reasons"] == ["small sample held-out"]


def test_adaptation_summary_none_when_no_plan_supplied():
    from omegaprompt.runtime import _build_adaptation_summary

    assert _build_adaptation_summary(None) is None


def test_applied_parameters_set_matches_runtime_consumers():
    """Drift guard: every name in APPLIED_PARAMETERS must actually be
    referenced in runtime.py (otherwise the set lies). Cheap regex check
    against the source so the next person who wires a new override into
    runtime updates the set in the same change.
    """
    import inspect
    import omegaprompt.runtime as runtime_mod

    src = inspect.getsource(runtime_mod)
    for param in APPLIED_PARAMETERS:
        # A consumer signal: runtime threads the value through tuning/resolved
        # locals, passes it as a kwarg to apply_adaptation_plan, or (for
        # manual_review) processes it via the ship-gate escalation path.
        readable_forms = [
            f"tuning.{param}",
            f"resolved_{param}",
            f"{param}=resolved_",
            f'"{param}"',
            f"'{param}'",
        ]
        if param == "manual_review":
            readable_forms += [
                "requires_manual_review",
                "require_manual_review",
                "manual_review_required",
                "manual_review_extras",
            ]
        assert any(form in src for form in readable_forms), (
            f"APPLIED_PARAMETERS contains {param!r} but runtime.py does not "
            f"reference any of {readable_forms}. Either the runtime stopped "
            "consuming this override or the set is stale."
        )


def test_advisory_overrides_have_no_runtime_consumer():
    """Inverse drift guard: parameters NOT in APPLIED_PARAMETERS but
    recorded by derive_adaptation_plan must NOT be consumed by runtime.
    If they are, they belong in APPLIED_PARAMETERS.

    Lists the advisory parameter names derive_adaptation_plan currently
    emits so a future change that wires one in is forced to update both
    sides at once.
    """
    import inspect
    import omegaprompt.runtime as runtime_mod

    src = inspect.getsource(runtime_mod)
    # Parameters derive_adaptation_plan emits as advisory (no runtime
    # consumer wired today). If runtime.py starts reading any of these,
    # move the name into APPLIED_PARAMETERS.
    advisory_names = {
        "rescore_count",
        "schema_mode",  # schema_mode_fallback in plan, but ParameterOverride.parameter is "schema_mode"
        "judge_ensemble_shift",
        "skip_axes",
        "rubric_weight_overrides",
        "candidate_budget_cap",
        "dataset_reorder_for_cache",
    }
    leaked = []
    for param in advisory_names:
        # Look for ``plan.{param}`` or ``adaptation_plan.{param}`` reads
        # in runtime — that would indicate runtime now consumes them.
        for prefix in ("plan.", "adaptation_plan."):
            if f"{prefix}{param}" in src:
                leaked.append(f"{prefix}{param}")
    assert leaked == [], (
        f"runtime.py reads advisory parameters {leaked}; move their "
        "names into APPLIED_PARAMETERS so the artifact summary stops "
        "marking them advisory_not_applied."
    )


def test_calibration_artifact_adaptation_summary_round_trips_through_json():
    from omegaprompt.domain import ShipRecommendation
    from omegaprompt.domain.result import CalibrationArtifact

    artifact = CalibrationArtifact(
        method="p1",
        unlock_k=3,
        best_params={"system_prompt_variant": 1},
        best_fitness=0.8,
        calibrated_params={"system_prompt_variant": 1},
        calibrated_fitness=0.8,
        hard_gate_pass_rate=1.0,
        n_candidates_evaluated=1,
        total_api_calls=1,
        ship_recommendation=ShipRecommendation.SHIP,
        adaptation_summary={
            "applied": ["min_kc4"],
            "advisory_not_applied": ["rescore_count"],
            "manual_review_required": False,
            "manual_review_reasons": [],
        },
    )
    rt = CalibrationArtifact.model_validate_json(artifact.model_dump_json())
    assert rt.adaptation_summary == artifact.adaptation_summary
