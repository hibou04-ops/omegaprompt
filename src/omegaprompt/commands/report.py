"""``omegaprompt report`` - render a CalibrationArtifact as Markdown.

A compact, human-readable summary intended for PR descriptions and CI
step outputs. The underlying artifact JSON remains the source of truth.
"""

from __future__ import annotations

from pathlib import Path

import typer

from omegaprompt.core.artifact import load_artifact
from omegaprompt.reporting import render_markdown


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
    md = render_markdown(artifact)
    if output_path is None:
        typer.echo(md)
    else:
        output_path.write_text(md, encoding="utf-8")
        typer.secho(f"Wrote {output_path}", fg=typer.colors.GREEN)
