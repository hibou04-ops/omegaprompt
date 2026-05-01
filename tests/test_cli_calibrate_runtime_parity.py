"""Reviewer P0: CLI ``omegaprompt calibrate`` delegates to runtime.calibrate.

Pre-fix the CLI re-implemented the pipeline. ``runtime.calibrate``
gained ``validation_mode``, ``adaptation_plan``, ``apply_adaptation_plan``,
``apply_ship_gate_escalation`` over time; the CLI never picked them up.
Three public surfaces (Python / CLI / MCP) had three different gate
policies.

Post-fix: the CLI is a thin Typer wrapper that builds CalibrateTuning
+ ProviderSpec from args and calls runtime.calibrate. New flags exposed:
- --validation-mode {auto, paired, disjoint}
- --adaptation-plan PATH
"""
from __future__ import annotations

from typer.testing import CliRunner

from omegaprompt.cli import app


runner = CliRunner()


def test_cli_calibrate_help_lists_validation_mode():
    result = runner.invoke(app, ["calibrate", "--help"], env={"COLUMNS": "200"})
    import re
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
    assert "--validation-mode" in plain
    assert "auto" in plain and "paired" in plain and "disjoint" in plain


def test_cli_calibrate_help_lists_adaptation_plan():
    result = runner.invoke(app, ["calibrate", "--help"], env={"COLUMNS": "200"})
    import re
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
    assert "--adaptation-plan" in plain


def test_cli_calibrate_rejects_unknown_validation_mode(tmp_path, monkeypatch):
    """Misspelling exits 2 with a clean error, not a confusing later
    failure inside runtime.calibrate."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    # Build a minimal valid set of input files
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        '{"id":"t1","input":"q","reference":"a"}\n'
        '{"id":"t2","input":"q","reference":"a"}\n',
        encoding="utf-8",
    )
    rubric = tmp_path / "rubric.json"
    rubric.write_text(
        '{"dimensions":[{"name":"acc","description":"x","weight":1.0}],'
        '"hard_gates":[]}',
        encoding="utf-8",
    )
    variants = tmp_path / "variants.json"
    variants.write_text(
        '{"system_prompts":["You are X."],"few_shot_examples":[]}',
        encoding="utf-8",
    )
    output = tmp_path / "out.json"

    result = runner.invoke(app, [
        "calibrate", str(dataset),
        "--rubric", str(rubric),
        "--variants", str(variants),
        "--output", str(output),
        "--validation-mode", "huh",
    ], catch_exceptions=False)

    assert result.exit_code == 2
    # Error message goes to stderr (typer.secho err=True). Combined
    # output via .output covers both streams.
    combined = (result.output or "") + (result.stderr if hasattr(result, "stderr") else "")
    assert "validation-mode" in combined or "validation_mode" in combined


def test_cli_calibrate_imports_runtime_not_local_pipeline():
    """Smoke-check that the CLI module no longer imports run_p1 directly.
    If a future refactor accidentally re-introduces a parallel pipeline,
    this test catches it. The runtime is the canonical owner of the
    p1 pipeline; the CLI's only job is argument parsing + delegation."""
    from omegaprompt.commands import calibrate as calibrate_mod
    import inspect

    source = inspect.getsource(calibrate_mod)
    # No direct run_p1 call from the CLI module:
    assert "run_p1(" not in source
    # The CLI must import runtime_calibrate:
    assert "from omegaprompt.runtime import" in source
    assert "calibrate as runtime_calibrate" in source
