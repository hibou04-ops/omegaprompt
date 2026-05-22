"""``omegaprompt diff`` — compare two calibration artifacts.

Reviewer P0: pre-fix this CLI command had its own regression logic
that diverged from ``runtime.diff()``. The runtime treats
``new.status != "OK"`` and ``ship_recommendation in {BLOCK, HOLD}``
as regressions; the CLI didn't, so the same artifact pair could be
\"OK\" in one surface and \"REGRESSION\" in the other.

This module is now a thin Typer wrapper around ``runtime.diff()`` so
all three public surfaces (Python runtime / CLI / MCP) share one
canonical regression contract.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
import typer

from omegaprompt.runtime import diff as runtime_diff


def diff(
    old_path: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path to the OLD CalibrationArtifact JSON (baseline).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    new_path: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path to the NEW CalibrationArtifact JSON (candidate).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    fail_on_regression: bool = typer.Option(  # noqa: B008
        True,
        "--fail-on-regression/--no-fail-on-regression",
        help=(
            "Exit 1 when the new artifact regresses on metrics OR carries a "
            "non-OK status / ship_recommendation in {BLOCK, HOLD}. "
            "Matches `runtime.diff()`."
        ),
    ),
) -> None:
    """Diff two calibration artifacts."""
    # Markdown form is the human-readable view.
    try:
        md = runtime_diff(old_path, new_path, format="markdown")
        structured = runtime_diff(old_path, new_path, format="json")
    except (ValidationError, ValueError) as exc:
        typer.secho(
            "INVALID_ARTIFACT: one or both inputs failed CalibrationArtifact validation.",
            fg=typer.colors.RED,
            err=True,
        )
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(md)

    # Structured form drives the exit code so CI gates honour the same
    # regression definition as Python callers.
    if structured.regressed and fail_on_regression:
        raise typer.Exit(code=1)
