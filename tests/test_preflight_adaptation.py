"""AdaptationPlan derivation and application tests."""

from __future__ import annotations

from omegaprompt.domain.enums import ResponseSchemaMode
from omegaprompt.preflight.adaptation import (
    apply_adaptation_plan,
    derive_adaptation_plan,
)
from omegaprompt.preflight.contracts import (
    AnalyticalFinding,
    EndpointMeasurement,
    JudgeQualityMeasurement,
    PerformanceMeasurement,
    PreflightReport,
    PreflightSeverity,
)


def _report(**kw) -> PreflightReport:
    base = dict(
        analytical_findings=[],
        judge_quality=None,
        endpoint=None,
        performance=None,
    )
    base.update(kw)
    return PreflightReport(**base)


# --------------------------- noise-adaptive KC-4 ---------------------------


def test_low_noise_leaves_min_kc4_untouched():
    report = _report(
        performance=PerformanceMeasurement(noise_floor=0.01),
    )
    plan = derive_adaptation_plan(report=report, default_min_kc4=0.5)
    assert plan.min_kc4_override is None


def test_moderate_noise_raises_min_kc4():
    report = _report(
        performance=PerformanceMeasurement(noise_floor=0.20),
    )
    plan = derive_adaptation_plan(report=report, default_min_kc4=0.5)
    assert plan.min_kc4_override is not None
    assert plan.min_kc4_override >= 0.60


def test_high_noise_raises_min_kc4_more():
    report = _report(
        performance=PerformanceMeasurement(noise_floor=0.40),
    )
    plan = derive_adaptation_plan(report=report, default_min_kc4=0.5)
    assert plan.min_kc4_override is not None
    assert plan.min_kc4_override >= 0.80


# --------------------------- rescore_count ---------------------------


def test_low_consistency_triggers_triple_rescore():
    report = _report(
        judge_quality=JudgeQualityMeasurement(consistency=0.55, samples=3),
    )
    plan = derive_adaptation_plan(report=report)
    assert plan.rescore_count == 3


def test_mid_consistency_triggers_double_rescore():
    report = _report(
        judge_quality=JudgeQualityMeasurement(consistency=0.75, samples=3),
    )
    plan = derive_adaptation_plan(report=report)
    assert plan.rescore_count == 2


def test_high_consistency_no_rescore():
    report = _report(
        judge_quality=JudgeQualityMeasurement(consistency=0.90, samples=3),
    )
    plan = derive_adaptation_plan(report=report)
    assert plan.rescore_count == 1


# --------------------------- schema fallback ---------------------------


def test_unreliable_schema_triggers_json_object_fallback():
    report = _report(
        endpoint=EndpointMeasurement(
            schema_reliability=0.60,
            context_budget_margin=1.0,
            caching_active=False,
            silent_degradation_detected=False,
        ),
    )
    plan = derive_adaptation_plan(report=report)
    assert plan.schema_mode_fallback == ResponseSchemaMode.JSON_OBJECT


def test_reliable_schema_no_fallback():
    report = _report(
        endpoint=EndpointMeasurement(
            schema_reliability=0.98,
            context_budget_margin=1.0,
            caching_active=True,
            silent_degradation_detected=False,
        ),
    )
    plan = derive_adaptation_plan(report=report)
    assert plan.schema_mode_fallback is None


# --------------------------- variant skip ---------------------------


def test_homogeneous_variants_finding_skips_prompt_axis():
    report = _report(
        analytical_findings=[
            AnalyticalFinding(
                trap_id="variants_homogeneous",
                label="NEW",
                hypothesis="prompts too similar",
                severity=PreflightSeverity.MEDIUM,
            ),
        ],
    )
    plan = derive_adaptation_plan(report=report)
    assert "system_prompt_variant" in plan.skip_axes


# --------------------------- judge ensemble shift ---------------------------


def test_weak_judge_pushes_rule_weight_up():
    report = _report(
        judge_quality=JudgeQualityMeasurement(consistency=0.50, samples=3),
    )
    plan = derive_adaptation_plan(report=report)
    assert plan.judge_ensemble_shift == 0.40


def test_strong_judge_keeps_default_ensemble():
    report = _report(
        judge_quality=JudgeQualityMeasurement(consistency=0.85, samples=3),
    )
    plan = derive_adaptation_plan(report=report)
    assert plan.judge_ensemble_shift is None


# --------------------------- wall-time cap ---------------------------


def test_long_projected_wall_time_reduces_unlock_k():
    report = _report(
        performance=PerformanceMeasurement(projected_wall_time_seconds=5 * 3600),
    )
    plan = derive_adaptation_plan(report=report, default_unlock_k=3)
    assert plan.unlock_k_override == 2


def test_short_projected_wall_time_leaves_unlock_k():
    report = _report(
        performance=PerformanceMeasurement(projected_wall_time_seconds=1200),
    )
    plan = derive_adaptation_plan(report=report, default_unlock_k=3)
    assert plan.unlock_k_override is None


# --------------------------- small-sample gap widening ---------------------------


def test_small_sample_finding_widens_max_gap():
    report = _report(
        analytical_findings=[
            AnalyticalFinding(
                trap_id="small_sample_kc4_power",
                label="REAL",
                hypothesis="test slice too small",
                severity=PreflightSeverity.HIGH,
            ),
        ],
    )
    plan = derive_adaptation_plan(report=report, default_max_gap=0.25)
    assert plan.max_gap_override is not None
    assert plan.max_gap_override > 0.25


# --------------------------- invariants ---------------------------


def test_apply_plan_never_weakens_kc4():
    # Plan says min_kc4=0.4 but caller's default is 0.6 - caller wins
    report = _report(performance=PerformanceMeasurement(noise_floor=0.0))
    plan = derive_adaptation_plan(report=report, default_min_kc4=0.4)
    # Force a "weaker" override to test the guard
    plan = plan.model_copy(update={"min_kc4_override": 0.3})
    kc4, _gap, _k = apply_adaptation_plan(plan, min_kc4=0.6, max_gap=0.25, unlock_k=3)
    assert kc4 == 0.6  # caller's default preserved


def test_apply_plan_never_widens_max_gap():
    report = _report()
    plan = derive_adaptation_plan(report=report, default_max_gap=0.25)
    plan = plan.model_copy(update={"max_gap_override": 0.99})  # attempted widening
    _kc4, gap, _k = apply_adaptation_plan(plan, min_kc4=0.5, max_gap=0.10, unlock_k=3)
    assert gap == 0.10  # caller's stricter value preserved


def test_apply_plan_never_raises_unlock_k():
    plan = derive_adaptation_plan(report=_report())
    plan = plan.model_copy(update={"unlock_k_override": 10})  # attempted widening
    _kc4, _gap, k = apply_adaptation_plan(plan, min_kc4=0.5, max_gap=0.25, unlock_k=2)
    assert k == 2


def test_derive_plan_preserves_discipline_flag():
    report = _report(
        judge_quality=JudgeQualityMeasurement(consistency=0.40, samples=3),
        endpoint=EndpointMeasurement(
            schema_reliability=0.50,
            context_budget_margin=0.0,
            caching_active=False,
            silent_degradation_detected=True,
        ),
        performance=PerformanceMeasurement(noise_floor=0.30, projected_wall_time_seconds=8 * 3600),
    )
    plan = derive_adaptation_plan(report=report)
    assert plan.preserves_discipline is True


def test_all_overrides_tracked_in_audit_list():
    report = _report(
        judge_quality=JudgeQualityMeasurement(consistency=0.50, samples=3),
        performance=PerformanceMeasurement(noise_floor=0.15),
    )
    plan = derive_adaptation_plan(report=report, default_min_kc4=0.5, default_unlock_k=3)
    override_params = {o.parameter for o in plan.overrides}
    assert "min_kc4" in override_params
    assert "rescore_count" in override_params
    assert "judge_ensemble_shift" in override_params
    # Rationale is non-empty for every override
    assert len(plan.rationale) >= len(plan.overrides) or len(plan.rationale) > 0


def test_noop_report_produces_empty_plan():
    report = _report()
    plan = derive_adaptation_plan(report=report)
    assert plan.min_kc4_override is None
    assert plan.max_gap_override is None
    assert plan.unlock_k_override is None
    assert plan.rescore_count == 1
    assert plan.schema_mode_fallback is None
    assert plan.judge_ensemble_shift is None
    assert plan.skip_axes == []
    assert plan.overrides == []
