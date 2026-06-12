"""``omegaprompt gate`` — first-class CI ship gate.

Runs the zero-network artifact integrity audit *and* the holdout
transfer/gap (overfit) verdict, then exits 0 (clear to ship) or non-zero
(blocked). A dedicated ship gate so CI no longer has to infer shippability
from ``diff`` / ``report``.

Exit codes:

* ``0`` — gate passed (integrity valid, release-approved, generalized).
* ``1`` — a ship-blocking condition (integrity error, not release-approved,
  overfit / unmeasured generalization).
* ``2`` — environment/load failure (artifact missing or unreadable), or
  invalid CLI arguments.
"""

from __future__ import annotations

from pathlib import Path

import typer

from omegaprompt.core.gate import render_gate_report, run_gate


def gate(
    artifact_path: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path to a CalibrationArtifact JSON to gate.",
    ),
    fmt: str = typer.Option(  # noqa: B008
        "text",
        "--format",
        "-f",
        help="Output format: text (default, human-readable) or json (machine summary).",
    ),
    require_generalization: bool = typer.Option(  # noqa: B008
        True,
        "--require-generalization/--no-require-generalization",
        help=(
            "When set (default), an absent or unverifiable walk-forward "
            "verdict blocks the gate. Disable to gate on integrity + "
            "release-approval only (the walk-forward read is still reported)."
        ),
    ),
) -> None:
    """Decide whether a calibration artifact is clear to ship."""
    fmt_norm = fmt.lower().strip()
    if fmt_norm not in {"text", "json"}:
        typer.secho(
            f"INVALID_FORMAT: {fmt!r} (expected text or json).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    result = run_gate(artifact_path, require_generalization=require_generalization)

    if fmt_norm == "json":
        import json

        typer.echo(
            json.dumps(
                result.to_json_dict(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
    else:
        typer.echo(render_gate_report(result))

    if result.exit_code != 0:
        raise typer.Exit(code=result.exit_code)
