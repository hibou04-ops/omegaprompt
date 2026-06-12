"""``omegaprompt report`` - render a CalibrationArtifact in several formats.

A compact, human-readable summary intended for PR descriptions and CI
step outputs. The underlying artifact JSON remains the source of truth.

Formats:

* ``markdown`` (default) — the existing compact PR/CI summary.
* ``json`` — a stable, schema-versioned summary dict (CI-consumable;
  includes the prominent train<->holdout overfit block).
* ``html`` — a self-contained single-file scorecard (stdlib only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import ValidationError
import typer

from omegaprompt.core.artifact import load_artifact
from omegaprompt.reporting import render_html, render_markdown, render_summary_json


def report(
    artifact_path: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path to a CalibrationArtifact JSON.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    fmt: str = typer.Option(  # noqa: B008
        "markdown",
        "--format",
        "-f",
        help="Output format: markdown (default), json, or html.",
    ),
    output_path: Optional[Path] = typer.Option(  # noqa: B008
        None,
        "--output",
        "-o",
        help="Where to write the rendered report. Defaults to stdout.",
    ),
) -> None:
    """Render a calibration artifact as markdown, json, or html."""
    fmt_norm = fmt.lower().strip()
    if fmt_norm not in {"markdown", "json", "html"}:
        typer.secho(
            f"INVALID_FORMAT: {fmt!r} (expected markdown, json, or html).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        artifact = load_artifact(artifact_path)
    except (ValidationError, ValueError) as exc:
        typer.secho(
            f"INVALID_ARTIFACT: {artifact_path} failed CalibrationArtifact validation.",
            fg=typer.colors.RED,
            err=True,
        )
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    if fmt_norm == "json":
        rendered = render_summary_json(artifact)
    elif fmt_norm == "html":
        rendered = render_html(artifact)
    else:
        rendered = render_markdown(artifact)

    if output_path is None:
        typer.echo(rendered)
    else:
        output_path.write_text(rendered, encoding="utf-8")
        typer.secho(f"Wrote {output_path}", fg=typer.colors.GREEN)
