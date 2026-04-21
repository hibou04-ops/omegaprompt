"""Reproducible preflight demo — shows analytical + adaptation on a
deterministic config.

No API calls. The analytical preflight reads the config; the adaptation
layer derives a plan; we print + save the plan to JSON for inspection.

Run::

    python examples/reference/reproduce_preflight_demo.py

Output: examples/reference/reference_preflight_report.json
        examples/reference/reference_adaptation_plan.json
"""

from __future__ import annotations

import json
from pathlib import Path

from omegaprompt.domain.dataset import Dataset, DatasetItem
from omegaprompt.domain.judge import Dimension, HardGate, JudgeRubric
from omegaprompt.domain.params import PromptVariants
from omegaprompt.preflight import (
    AdaptationPlan,
    PreflightReport,
    analytical_preflight,
    derive_adaptation_plan,
)
from omegaprompt.preflight.contracts import (
    EndpointMeasurement,
    JudgeQualityMeasurement,
    PerformanceMeasurement,
)


def build_inputs():
    """Intentionally weak config to exercise the adaptation logic."""
    train = Dataset(items=[DatasetItem(id=f"t{i}", input=f"task {i}") for i in range(6)])
    test = Dataset(items=[DatasetItem(id=f"v{i}", input=f"val {i}") for i in range(5)])
    rubric = JudgeRubric(
        dimensions=[
            Dimension(name="accuracy", description="is it right", weight=0.85),
            Dimension(name="clarity",  description="is it readable", weight=0.15),
        ],
        hard_gates=[
            HardGate(name="no_refusal", description="model attempts", evaluator="judge"),
        ],
    )
    variants = PromptVariants(
        system_prompts=[
            "You are an assistant.",
            "You are an assistant helping.",
        ],
        few_shot_examples=[],
    )
    return train, test, rubric, variants


def main():
    train, test, rubric, variants = build_inputs()

    # Same-vendor + small sample + homogeneous variants + concentrated rubric
    analytical_findings = analytical_preflight(
        target_provider="openai",
        target_model="gpt-4o-mini",
        judge_provider="openai",
        judge_model="gpt-4o-mini",
        train_dataset=train,
        test_dataset=test,
        rubric=rubric,
        variants=variants,
        judge_output_budget="small",
    )

    # Simulated empirical measurements a weak endpoint might produce
    judge_quality = JudgeQualityMeasurement(consistency=0.55, anchoring_usage=0.4, samples=3)
    endpoint = EndpointMeasurement(
        schema_reliability=0.67,
        context_budget_margin=0.35,
        caching_active=False,
        silent_degradation_detected=False,
    )
    performance = PerformanceMeasurement(
        mean_call_latency_ms=42_000.0,
        projected_wall_time_seconds=5.2 * 3600,
        noise_floor=0.18,
    )

    report = PreflightReport(
        analytical_findings=analytical_findings,
        judge_quality=judge_quality,
        endpoint=endpoint,
        performance=performance,
    )

    plan = derive_adaptation_plan(report=report, default_min_kc4=0.5, default_max_gap=0.25, default_unlock_k=3)

    report_path = Path(__file__).with_name("reference_preflight_report.json")
    plan_path   = Path(__file__).with_name("reference_adaptation_plan.json")
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    print(f"Wrote {report_path}")
    print(f"Wrote {plan_path}")
    print("\n=== Analytical findings ===")
    for f in report.analytical_findings:
        print(f"  [{f.label:<10} {f.severity.value:<7}] {f.trap_id}")
        if f.note:
            print(f"      note: {f.note}")
        if f.remediation:
            print(f"      remediation: {f.remediation}")

    print("\n=== Empirical measurements ===")
    print(f"  judge.consistency          = {report.judge_quality.consistency:.2f}")
    print(f"  judge.anchoring_usage      = {report.judge_quality.anchoring_usage:.2f}")
    print(f"  endpoint.schema_reliability = {report.endpoint.schema_reliability:.2f}")
    print(f"  endpoint.context_margin     = {report.endpoint.context_budget_margin:.2f}")
    print(f"  perf.projected_wall_time   = {report.performance.projected_wall_time_seconds/3600:.1f}h")
    print(f"  perf.noise_floor           = {report.performance.noise_floor:.3f}")

    print("\n=== AdaptationPlan ===")
    for o in plan.overrides:
        print(f"  {o.parameter:<20} {str(o.default):<8} -> {str(o.applied):<8}  ({o.reason})")
    print(f"\n  rescore_count           = {plan.rescore_count}")
    print(f"  skip_axes               = {plan.skip_axes}")
    print(f"  schema_mode_fallback    = {plan.schema_mode_fallback}")
    print(f"  judge_ensemble_shift    = {plan.judge_ensemble_shift}")
    print(f"  preserves_discipline    = {plan.preserves_discipline}")


if __name__ == "__main__":
    main()
