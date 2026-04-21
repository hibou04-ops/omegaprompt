"""``omegacal diff`` - compare two calibration artifacts.

Intended for CI: catch regressions when someone edits the prompt or
variants. Exit code reflects whether the new artifact is strictly
better or at least not worse on the hard metrics.
"""

from __future__ import annotations

from pathlib import Path

import typer

from omegaprompt.core.artifact import load_artifact


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
        help="Exit 1 when the new artifact regresses on fitness or walk-forward.",
    ),
) -> None:
    """Diff two calibration artifacts."""
    old = load_artifact(old_path)
    new = load_artifact(new_path)

    def _arrow(a: float, b: float) -> str:
        if a == b:
            return "="
        return "up" if b > a else "down"

    typer.echo(f"# omegacal diff\n")
    typer.echo(f"OLD: {old_path}  status={old.status}  calibrated={old.calibrated_fitness:.4f}")
    typer.echo(f"NEW: {new_path}  status={new.status}  calibrated={new.calibrated_fitness:.4f}")
    typer.echo("")
    typer.echo(
        f"- calibrated_fitness: {old.calibrated_fitness:.4f} -> {new.calibrated_fitness:.4f}  "
        f"[{_arrow(old.calibrated_fitness, new.calibrated_fitness)}]"
    )
    typer.echo(
        f"- neutral_fitness: {old.neutral_fitness:.4f} -> {new.neutral_fitness:.4f}  "
        f"[{_arrow(old.neutral_fitness, new.neutral_fitness)}]"
    )

    regressed = False

    if new.calibrated_fitness < old.calibrated_fitness:
        regressed = True
    if new.quality_per_cost_best < old.quality_per_cost_best:
        regressed = True
    if new.quality_per_latency_best < old.quality_per_latency_best:
        regressed = True

    if old.walk_forward is not None and new.walk_forward is not None:
        typer.echo(
            f"- test_fitness: {old.walk_forward.test_fitness:.4f} -> "
            f"{new.walk_forward.test_fitness:.4f}  "
            f"[{_arrow(old.walk_forward.test_fitness, new.walk_forward.test_fitness)}]"
        )
        typer.echo(
            f"- gen_gap: {old.walk_forward.generalization_gap:.2%} -> "
            f"{new.walk_forward.generalization_gap:.2%}"
        )
        if new.walk_forward.test_fitness < old.walk_forward.test_fitness:
            regressed = True
        if old.walk_forward.passed and not new.walk_forward.passed:
            regressed = True

    typer.echo(
        f"- hard_gate_pass_rate: {old.hard_gate_pass_rate:.1%} -> "
        f"{new.hard_gate_pass_rate:.1%}"
    )
    if new.hard_gate_pass_rate < old.hard_gate_pass_rate:
        regressed = True
    typer.echo(
        f"- stayed_within_guarded_boundaries: {old.stayed_within_guarded_boundaries} -> "
        f"{new.stayed_within_guarded_boundaries}"
    )
    if old.stayed_within_guarded_boundaries and not new.stayed_within_guarded_boundaries:
        regressed = True

    if regressed and fail_on_regression:
        typer.secho("REGRESSION detected.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.secho("OK", fg=typer.colors.GREEN)
