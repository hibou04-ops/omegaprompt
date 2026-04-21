"""``omegaprompt report`` - render a CalibrationArtifact as Markdown.

A compact, human-readable summary intended for PR descriptions and CI
step outputs. The underlying artifact JSON remains the source of truth.
"""

from __future__ import annotations

from pathlib import Path

import typer

from omegaprompt.core.artifact import load_artifact


def report(
    artifact_path: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path to a CalibrationArtifact JSON.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--output",
        "-o",
        help="Where to write the Markdown. Defaults to stdout.",
    ),
) -> None:
    """Render a calibration artifact as Markdown."""
    artifact = load_artifact(artifact_path)
    md = _render_markdown(artifact)
    if output_path is None:
        typer.echo(md)
    else:
        output_path.write_text(md, encoding="utf-8")
        typer.secho(f"Wrote {output_path}", fg=typer.colors.GREEN)


def _render_markdown(a) -> str:
    lines: list[str] = []
    lines.append(f"# omegaprompt calibration - {a.status}")
    lines.append("")
    lines.append(f"- **Method:** {a.method} (unlock_k={a.unlock_k})")
    lines.append(f"- **Train fitness:** {a.best_fitness:.4f}")
    if a.walk_forward is not None:
        wf = a.walk_forward
        lines.append(f"- **Test fitness:** {wf.test_fitness:.4f}")
        lines.append(
            f"- **Generalization gap:** {wf.generalization_gap:.2%}"
            + (f" (KC-4 r={wf.kc4_correlation:.3f})" if wf.kc4_correlation is not None else "")
            + (" PASS" if wf.passed else " FAIL")
        )
    lines.append(f"- **Hard-gate pass rate:** {a.hard_gate_pass_rate:.1%}")
    lines.append(
        f"- **Target:** {a.target_provider or '-'}/{a.target_model or '-'}  "
        f"**Judge:** {a.judge_provider or '-'}/{a.judge_model or '-'}"
    )
    if a.rationale and a.rationale != "passed":
        lines.append(f"- **Rationale:** {a.rationale}")
    lines.append("")
    lines.append("## Best parameters")
    lines.append("")
    lines.append("```json")
    lines.append(_pretty_json(a.best_params))
    lines.append("```")
    lines.append("")
    if a.sensitivity_ranking:
        lines.append("## Sensitivity ranking")
        lines.append("")
        lines.append("| rank | axis | gini_delta |")
        lines.append("| --- | --- | --- |")
        for row in a.sensitivity_ranking:
            gd = row.get("gini_delta")
            gd_s = f"{gd:.3f}" if isinstance(gd, (int, float)) else str(gd)
            lines.append(f"| {row.get('rank')} | {row.get('axis')} | {gd_s} |")
        lines.append("")
    if a.usage_summary:
        lines.append("## Token usage")
        lines.append("")
        for k, v in a.usage_summary.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    lines.append(f"- total_api_calls: {a.total_api_calls}")
    lines.append(f"- n_candidates_evaluated: {a.n_candidates_evaluated}")
    return "\n".join(lines).rstrip() + "\n"


def _pretty_json(obj) -> str:
    import json

    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
