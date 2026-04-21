"""Top-level Typer application for omegacal."""

import typer

from omegaprompt import __version__
from omegaprompt.commands import calibrate as calibrate_cmd
from omegaprompt.commands import diff as diff_cmd
from omegaprompt.commands import report as report_cmd

app = typer.Typer(
    name="omegacal",
    help="Provider-neutral prompt calibration engine with walk-forward ship gates.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"omegacal {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(  # noqa: B008
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """omegacal - provider-neutral prompt calibration with structural risk reporting."""


app.command(
    name="calibrate",
    help="Calibrate a prompt configuration (stress + grid + walk-forward).",
)(calibrate_cmd.calibrate)

app.command(
    name="report",
    help="Render a CalibrationArtifact JSON as Markdown.",
)(report_cmd.report)

app.command(
    name="diff",
    help="Compare two CalibrationArtifact JSONs (for CI regression detection).",
)(diff_cmd.diff)


if __name__ == "__main__":
    app()
