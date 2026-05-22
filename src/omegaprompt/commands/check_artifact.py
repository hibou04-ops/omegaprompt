"""``omegaprompt check-artifact`` — zero-network artifact integrity audit."""

from __future__ import annotations

from pathlib import Path

import typer

from omegaprompt.core.artifact_integrity import (
    check_artifact_integrity,
    render_integrity_report,
)


def check_artifact(
    artifact_path: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path to a CalibrationArtifact JSON.",
    ),
    strict: bool = typer.Option(  # noqa: B008
        False,
        "--strict",
        help="Exit non-zero when integrity errors are found.",
    ),
    json_output: bool = typer.Option(  # noqa: B008
        False,
        "--json",
        help="Emit the machine-readable JSON report instead of the human report.",
    ),
) -> None:
    """Check a CalibrationArtifact without network or provider calls."""

    report = check_artifact_integrity(artifact_path)
    if json_output:
        typer.echo(report.model_dump_json(indent=2))
    else:
        typer.echo(render_integrity_report(report))

    if report.environment_blocked:
        raise typer.Exit(code=2)
    if strict and report.strict_blocking_findings:
        raise typer.Exit(code=1)
