"""Markdown rendering for calibration artifacts."""

from __future__ import annotations

import json


def render_markdown(a) -> str:
    lines: list[str] = []
    lines.append(f"# omegacal calibration - {a.status}")
    lines.append("")
    lines.append(f"- **Profile:** {a.selected_profile.value}")
    lines.append(f"- **Method:** {a.method} (unlock_k={a.unlock_k})")
    lines.append(f"- **Neutral fitness:** {a.neutral_fitness:.4f}")
    lines.append(f"- **Calibrated fitness:** {a.calibrated_fitness:.4f}")
    lines.append(f"- **Uplift:** {a.uplift_absolute:+.4f} ({a.uplift_percent:+.2f}%)")
    lines.append(
        f"- **Quality per cost:** {a.quality_per_cost_neutral:.6f} -> {a.quality_per_cost_best:.6f}"
    )
    lines.append(
        f"- **Quality per latency:** {a.quality_per_latency_neutral:.6f} -> {a.quality_per_latency_best:.6f}"
    )
    if a.walk_forward is not None:
        wf = a.walk_forward
        lines.append(f"- **Test fitness:** {wf.test_fitness:.4f}")
        lines.append(
            f"- **Walk-forward:** gap {wf.generalization_gap:.2%}"
            + (f", KC-4 r={wf.kc4_correlation:.3f}" if wf.kc4_correlation is not None else "")
            + (" PASS" if wf.passed else " FAIL")
        )
    lines.append(f"- **Ship recommendation:** {a.ship_recommendation.value}")
    lines.append(f"- **Stayed within guarded boundaries:** {a.stayed_within_guarded_boundaries}")
    lines.append(
        f"- **Boundary-crossing uplift:** {a.additional_uplift_from_boundary_crossing:+.4f}"
    )
    lines.append(
        f"- **Target:** {a.target_provider or '-'}/{a.target_model or '-'}  "
        f"**Judge:** {a.judge_provider or '-'}/{a.judge_model or '-'}"
    )
    lines.append("")
    lines.append("## Calibrated Parameters")
    lines.append("")
    lines.append("```json")
    lines.append(_pretty_json(a.calibrated_params))
    lines.append("```")
    lines.append("")
    lines.append("## Neutral Baseline")
    lines.append("")
    lines.append("```json")
    lines.append(_pretty_json(a.neutral_baseline_params))
    lines.append("```")
    if a.boundary_warnings:
        lines.append("")
        lines.append("## Boundary Warnings")
        lines.append("")
        for warning in a.boundary_warnings:
            lines.append(
                f"- **{warning.category.value} / {warning.severity}**: {warning.summary} "
                f"({warning.detail})"
            )
    if a.degraded_capabilities:
        lines.append("")
        lines.append("## Degraded Capabilities")
        lines.append("")
        for event in a.degraded_capabilities:
            lines.append(
                f"- `{event.capability}`: {event.requested} -> {event.applied}. {event.user_visible_note}"
            )
    if a.relaxed_safeguards:
        lines.append("")
        lines.append("## Relaxed Safeguards")
        lines.append("")
        for safeguard in a.relaxed_safeguards:
            lines.append(
                f"- `{safeguard.name}`: {safeguard.reason} Risk: {safeguard.increased_risk}"
            )
    if a.sensitivity_ranking:
        lines.append("")
        lines.append("## Sensitivity Ranking")
        lines.append("")
        lines.append("| rank | axis | gini_delta |")
        lines.append("| --- | --- | --- |")
        for row in a.sensitivity_ranking:
            gd = row.get("gini_delta")
            gd_s = f"{gd:.3f}" if isinstance(gd, (int, float)) else str(gd)
            lines.append(f"| {row.get('rank')} | {row.get('axis')} | {gd_s} |")
    lines.append("")
    lines.append("## Run Metrics")
    lines.append("")
    lines.append(f"- hard_gate_pass_rate: {a.hard_gate_pass_rate:.1%}")
    lines.append(f"- n_candidates_evaluated: {a.n_candidates_evaluated}")
    lines.append(f"- total_api_calls: {a.total_api_calls}")
    if a.usage_summary:
        for key, value in a.usage_summary.items():
            lines.append(f"- {key}: {value}")
    return "\n".join(lines).rstrip() + "\n"


def _pretty_json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
