"""Reproducible preflight-consumer demo.

Shows how to hand-construct a :class:`PreflightReport` and feed it through
:func:`derive_adaptation_plan` to get an :class:`AdaptationPlan`. In a
production workflow, a sub-tool such as ``mini-omega-lock`` or
``mini-antemortem-cli`` (separate packages, separate repositories) would
produce the report for you; this demo mirrors what such a tool's output
would look like, so the adaptation logic can be exercised offline.

Run::

    python examples/reference/reproduce_preflight_demo.py

Output: examples/reference/reference_preflight_report.json
        examples/reference/reference_adaptation_plan.json
"""

from __future__ import annotations

from pathlib import Path

from omegaprompt.preflight import (
    AdaptationPlan,
    AnalyticalFinding,
    EndpointMeasurement,
    JudgeQualityMeasurement,
    PerformanceMeasurement,
    PreflightReport,
    PreflightSeverity,
    derive_adaptation_plan,
)


def _hand_built_report() -> PreflightReport:
    """What a sub-tool-emitted report *would* look like on a weak setup.

    The findings below are the kind of classifications a ``mini-antemortem-cli``
    run against the same configuration would produce; the measurements are the
    kind of numbers a ``mini-omega-lock`` probe set would record.
    """
    return PreflightReport(
        analytical_findings=[
            AnalyticalFinding(
                trap_id="self_agreement_bias",
                label="REAL",
                hypothesis="Target and judge share a vendor.",
                severity=PreflightSeverity.HIGH,
                note="Target and judge are identical: openai/gpt-4o-mini.",
                remediation="Use a cross-vendor judge.",
            ),
            AnalyticalFinding(
                trap_id="small_sample_kc4_power",
                label="REAL",
                hypothesis="Pearson power insufficient.",
                severity=PreflightSeverity.HIGH,
                note="Test slice has 5 items. Pearson at n=5 is underpowered.",
                remediation="Expand test slice or raise --min-kc4 adaptively.",
            ),
            AnalyticalFinding(
                trap_id="variants_homogeneous",
                label="NEW",
                hypothesis="Variants lack diversity.",
                severity=PreflightSeverity.MEDIUM,
                note="All system prompts are within 8 characters of each other.",
                remediation="Vary role framing and length more aggressively.",
            ),
            AnalyticalFinding(
                trap_id="rubric_weight_concentration",
                label="REAL",
                hypothesis="Single dimension dominates weight.",
                severity=PreflightSeverity.MEDIUM,
                note="Dimension 'accuracy' at 85%; judge noise on it dominates fitness.",
                remediation="Rebalance or declare intent.",
            ),
        ],
        judge_quality=JudgeQualityMeasurement(consistency=0.55, anchoring_usage=0.40, samples=3),
        endpoint=EndpointMeasurement(
            schema_reliability=0.67,
            context_budget_margin=0.35,
            caching_active=False,
            silent_degradation_detected=False,
        ),
        performance=PerformanceMeasurement(
            mean_call_latency_ms=42_000.0,
            projected_wall_time_seconds=5.2 * 3600,
            noise_floor=0.18,
        ),
    )


def main() -> None:
    report = _hand_built_report()
    plan: AdaptationPlan = derive_adaptation_plan(
        report=report,
        default_min_kc4=0.5,
        default_max_gap=0.25,
        default_unlock_k=3,
    )

    report_path = Path(__file__).with_name("reference_preflight_report.json")
    plan_path = Path(__file__).with_name("reference_adaptation_plan.json")
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    print(f"Wrote {report_path}")
    print(f"Wrote {plan_path}")
    print("\n=== Analytical findings (as a sub-tool would emit them) ===")
    for f in report.analytical_findings:
        print(f"  [{f.label:<10} {f.severity.value:<7}] {f.trap_id}")
        if f.note:
            print(f"      note: {f.note}")

    print("\n=== Empirical measurements ===")
    print(f"  judge.consistency           = {report.judge_quality.consistency:.2f}")
    print(f"  endpoint.schema_reliability = {report.endpoint.schema_reliability:.2f}")
    print(f"  perf.projected_wall_time    = {report.performance.projected_wall_time_seconds / 3600:.1f}h")
    print(f"  perf.noise_floor            = {report.performance.noise_floor:.3f}")

    print("\n=== AdaptationPlan overrides ===")
    for o in plan.overrides:
        print(f"  {o.parameter:<22} {str(o.default):<8} -> {str(o.applied):<16}  ({o.reason})")
    print(f"\n  rescore_count            = {plan.rescore_count}")
    print(f"  skip_axes                = {plan.skip_axes}")
    print(f"  schema_mode_fallback     = {plan.schema_mode_fallback}")
    print(f"  judge_ensemble_shift     = {plan.judge_ensemble_shift}")
    print(f"  preserves_discipline     = {plan.preserves_discipline}")


if __name__ == "__main__":
    main()
