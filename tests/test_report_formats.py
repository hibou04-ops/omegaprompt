"""Tests for the report/diff output formats added in 2.1.0."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from omegaprompt.cli import app
from omegaprompt.core.artifact import load_artifact
from omegaprompt.reporting import (
    REPORT_SUMMARY_SCHEMA_VERSION,
    build_report_summary,
    render_html,
    render_markdown,
    render_summary_json,
)

runner = CliRunner()

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "examples" / "reference"
CLEAN = REF / "reference_artifact.json"
DIFF_REGRESS = REF / "reference_diff_regression.json"


# --------------------------- report json -----------------------------


def test_build_report_summary_has_overfit_block() -> None:
    a = load_artifact(CLEAN)
    summary = build_report_summary(a)
    assert summary["summary_schema_version"] == REPORT_SUMMARY_SCHEMA_VERSION
    assert summary["artifact_schema_version"] == "2.0"
    assert summary["status"] == "OK"
    assert summary["ship_recommendation"] == "ship"
    assert "overfit" in summary
    assert summary["overfit"]["overfit_verdict"] == "GENERALIZES"


def test_render_summary_json_is_deterministic() -> None:
    a = load_artifact(CLEAN)
    assert render_summary_json(a) == render_summary_json(a)


def test_cli_report_json_format() -> None:
    result = runner.invoke(app, ["report", str(CLEAN), "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "OK"
    assert payload["overfit"]["overfit_verdict"] == "GENERALIZES"


def test_cli_report_markdown_is_default() -> None:
    default = runner.invoke(app, ["report", str(CLEAN)])
    explicit = runner.invoke(app, ["report", str(CLEAN), "--format", "markdown"])
    assert default.exit_code == 0
    assert default.stdout == explicit.stdout
    assert "# omegaprompt calibration" in default.stdout


def test_cli_report_invalid_format_exits_two() -> None:
    result = runner.invoke(app, ["report", str(CLEAN), "--format", "pdf"])
    assert result.exit_code == 2


# --------------------------- report html -----------------------------


def test_render_html_is_self_contained() -> None:
    a = load_artifact(CLEAN)
    html_doc = render_html(a)
    assert html_doc.startswith("<!doctype html>")
    assert "<style>" in html_doc  # inline CSS, no external assets
    assert "http://" not in html_doc.split("<style>")[0]  # no remote head links
    assert "<script" not in html_doc  # stdlib only, no JS
    assert "omegaprompt calibration scorecard" in html_doc
    assert "overfit:" in html_doc


def test_render_html_escapes_content() -> None:
    a = load_artifact(CLEAN)
    a.target_model = "<script>alert(1)</script>"
    html_doc = render_html(a)
    assert "<script>alert(1)</script>" not in html_doc
    assert "&lt;script&gt;" in html_doc


def test_cli_report_html_format(tmp_path: Path) -> None:
    out = tmp_path / "scorecard.html"
    result = runner.invoke(app, ["report", str(CLEAN), "--format", "html", "-o", str(out)])
    assert result.exit_code == 0
    content = out.read_text(encoding="utf-8")
    assert content.startswith("<!doctype html>")


# --------------------------- diff format -----------------------------


def test_cli_diff_json_format_parses() -> None:
    result = runner.invoke(
        app,
        ["diff", str(CLEAN), str(DIFF_REGRESS), "--format", "json", "--no-fail-on-regression"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["regressed"] is True
    assert payload["regression_reasons"]


def test_cli_diff_markdown_is_default() -> None:
    result = runner.invoke(
        app,
        ["diff", str(CLEAN), str(DIFF_REGRESS), "--no-fail-on-regression"],
    )
    assert result.exit_code == 0
    assert "# omegaprompt diff" in result.stdout


def test_cli_diff_json_still_drives_exit_code() -> None:
    result = runner.invoke(app, ["diff", str(CLEAN), str(DIFF_REGRESS), "--format", "json"])
    assert result.exit_code == 1


def test_cli_diff_invalid_format_exits_two() -> None:
    result = runner.invoke(app, ["diff", str(CLEAN), str(DIFF_REGRESS), "--format", "yaml"])
    assert result.exit_code == 2
