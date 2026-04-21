"""Smoke tests for the top-level CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from omegaprompt import __version__
from omegaprompt.cli import app

runner = CliRunner()


def test_help_lists_subcommands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "calibrate" in result.stdout
    assert "report" in result.stdout
    assert "diff" in result.stdout


def test_version_flag_prints_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_no_args_returns_nonzero_help():
    result = runner.invoke(app, [])
    assert result.exit_code != 0
    combined = result.stdout + result.output
    assert "Usage" in combined or "calibrate" in combined


def test_calibrate_help_documents_flags():
    result = runner.invoke(app, ["calibrate", "--help"])
    assert result.exit_code == 0
    for flag in ("--rubric", "--variants", "--target-model", "--judge-model"):
        assert flag in result.stdout


def test_report_renders_markdown(tmp_path: Path):
    artifact = {
        "schema_version": "1.0",
        "method": "p1",
        "unlock_k": 3,
        "best_params": {"system_prompt_idx": 1},
        "best_fitness": 0.85,
        "hard_gate_pass_rate": 1.0,
        "sensitivity_ranking": [
            {"axis": "system_prompt_idx", "gini_delta": 0.42, "rank": 0},
        ],
        "n_candidates_evaluated": 12,
        "total_api_calls": 40,
        "usage_summary": {"input_tokens": 500, "output_tokens": 200},
        "target_provider": "openai",
        "target_model": "gpt-4o",
        "judge_provider": "anthropic",
        "judge_model": "claude-opus-4-7",
        "status": "OK",
        "rationale": "passed",
    }
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    result = runner.invoke(app, ["report", str(artifact_path)])
    assert result.exit_code == 0
    assert "omegaprompt calibration" in result.stdout
    assert "0.8500" in result.stdout or "0.8" in result.stdout
    assert "system_prompt_idx" in result.stdout


def test_diff_detects_regression(tmp_path: Path):
    old_a = {
        "schema_version": "1.0",
        "method": "p1",
        "unlock_k": 3,
        "best_params": {},
        "best_fitness": 0.9,
        "hard_gate_pass_rate": 1.0,
        "n_candidates_evaluated": 1,
        "total_api_calls": 1,
        "status": "OK",
    }
    new_a = dict(old_a)
    new_a["best_fitness"] = 0.7
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    old_path.write_text(json.dumps(old_a), encoding="utf-8")
    new_path.write_text(json.dumps(new_a), encoding="utf-8")

    result = runner.invoke(app, ["diff", str(old_path), str(new_path)])
    assert result.exit_code == 1
    assert "REGRESSION" in result.stdout


def test_diff_passes_on_improvement(tmp_path: Path):
    old_a = {
        "schema_version": "1.0",
        "method": "p1",
        "unlock_k": 3,
        "best_params": {},
        "best_fitness": 0.7,
        "hard_gate_pass_rate": 0.8,
        "n_candidates_evaluated": 1,
        "total_api_calls": 1,
        "status": "OK",
    }
    new_a = dict(old_a)
    new_a["best_fitness"] = 0.85
    new_a["hard_gate_pass_rate"] = 0.9
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    old_path.write_text(json.dumps(old_a), encoding="utf-8")
    new_path.write_text(json.dumps(new_a), encoding="utf-8")

    result = runner.invoke(app, ["diff", str(old_path), str(new_path)])
    assert result.exit_code == 0
    assert "OK" in result.stdout
