"""Smoke tests for the top-level CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from omegaprompt import __version__
from omegaprompt.cli import app
from tests.helpers import workspace_tmpdir

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
    assert "omegaprompt" in result.stdout


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


def test_report_renders_markdown():
    artifact = {
        "schema_version": "2.0",
        "method": "p1",
        "unlock_k": 3,
        "best_params": {"system_prompt_variant": 1},
        "best_fitness": 0.85,
        "neutral_baseline_params": {"system_prompt_variant": 0},
        "neutral_fitness": 0.70,
        "calibrated_params": {"system_prompt_variant": 1},
        "calibrated_fitness": 0.85,
        "uplift_absolute": 0.15,
        "uplift_percent": 21.43,
        "quality_per_cost_neutral": 0.001,
        "quality_per_cost_best": 0.002,
        "quality_per_latency_neutral": 0.01,
        "quality_per_latency_best": 0.02,
        "hard_gate_pass_rate": 1.0,
        "sensitivity_ranking": [
            {"axis": "system_prompt_variant", "gini_delta": 0.42, "rank": 0},
        ],
        "n_candidates_evaluated": 12,
        "total_api_calls": 40,
        "usage_summary": {"input_tokens": 500, "output_tokens": 200},
        "target_provider": "openai",
        "target_model": "gpt-4o",
        "judge_provider": "anthropic",
        "judge_model": "claude-opus-4-7",
        "ship_recommendation": "ship",
        "stayed_within_guarded_boundaries": True,
        "additional_uplift_from_boundary_crossing": 0.0,
        "status": "OK",
        "rationale": "passed",
    }
    with workspace_tmpdir() as tmp_path:
        artifact_path = tmp_path / "artifact.json"
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

        result = runner.invoke(app, ["report", str(artifact_path)])
        assert result.exit_code == 0
        assert "omegaprompt calibration" in result.stdout
        assert "Calibrated fitness" in result.stdout
        assert "system_prompt_variant" in result.stdout


def test_diff_detects_regression():
    old_a = {
        "schema_version": "2.0",
        "method": "p1",
        "unlock_k": 3,
        "best_params": {},
        "best_fitness": 0.9,
        "neutral_fitness": 0.7,
        "calibrated_fitness": 0.9,
        "quality_per_cost_best": 0.2,
        "quality_per_latency_best": 0.1,
        "stayed_within_guarded_boundaries": True,
        "hard_gate_pass_rate": 1.0,
        "n_candidates_evaluated": 1,
        "total_api_calls": 1,
        "status": "OK",
    }
    new_a = dict(old_a)
    new_a["best_fitness"] = 0.7
    new_a["calibrated_fitness"] = 0.7
    new_a["quality_per_cost_best"] = 0.1
    with workspace_tmpdir() as tmp_path:
        old_path = tmp_path / "old.json"
        new_path = tmp_path / "new.json"
        old_path.write_text(json.dumps(old_a), encoding="utf-8")
        new_path.write_text(json.dumps(new_a), encoding="utf-8")

        result = runner.invoke(app, ["diff", str(old_path), str(new_path)])
        assert result.exit_code == 1
        assert "REGRESSION" in result.stdout


def test_diff_passes_on_improvement():
    old_a = {
        "schema_version": "2.0",
        "method": "p1",
        "unlock_k": 3,
        "best_params": {},
        "best_fitness": 0.7,
        "neutral_fitness": 0.6,
        "calibrated_fitness": 0.7,
        "quality_per_cost_best": 0.1,
        "quality_per_latency_best": 0.05,
        "stayed_within_guarded_boundaries": True,
        "hard_gate_pass_rate": 0.8,
        "n_candidates_evaluated": 1,
        "total_api_calls": 1,
        "status": "OK",
    }
    new_a = dict(old_a)
    new_a["best_fitness"] = 0.85
    new_a["calibrated_fitness"] = 0.85
    new_a["hard_gate_pass_rate"] = 0.9
    new_a["quality_per_cost_best"] = 0.2
    new_a["quality_per_latency_best"] = 0.06
    with workspace_tmpdir() as tmp_path:
        old_path = tmp_path / "old.json"
        new_path = tmp_path / "new.json"
        old_path.write_text(json.dumps(old_a), encoding="utf-8")
        new_path.write_text(json.dumps(new_a), encoding="utf-8")

        result = runner.invoke(app, ["diff", str(old_path), str(new_path)])
        assert result.exit_code == 0
        assert "OK" in result.stdout
