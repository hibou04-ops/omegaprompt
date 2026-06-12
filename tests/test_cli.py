"""Smoke tests for the top-level CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from omegaprompt import __version__
from omegaprompt.cli import app
from omegaprompt.domain import ShipRecommendation
from omegaprompt.domain.result import CalibrationArtifact
from tests.helpers import workspace_tmpdir

runner = CliRunner()


def _minimal_artifact(
    *,
    status: str = "OK",
    ship: ShipRecommendation = ShipRecommendation.SHIP,
    calibrated_fitness: float = 0.8,
) -> CalibrationArtifact:
    return CalibrationArtifact(
        schema_version="2.0",
        method="p1",
        unlock_k=1,
        best_params={"system_prompt_variant": 0},
        best_fitness=calibrated_fitness,
        calibrated_params={"system_prompt_variant": 0},
        calibrated_fitness=calibrated_fitness,
        neutral_fitness=0.7,
        quality_per_cost_best=0.2,
        quality_per_latency_best=0.1,
        hard_gate_pass_rate=1.0 if status != "FAIL_HARD_GATES" else 0.0,
        n_candidates_evaluated=1,
        total_api_calls=1,
        status=status,
        ship_recommendation=ship,
        rationale="test fixture",
    )


def _write_minimal_cli_inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    dataset = root / "train.jsonl"
    dataset.write_text('{"id":"t1","input":"q","reference":"a"}\n', encoding="utf-8")
    rubric = root / "rubric.json"
    rubric.write_text(
        '{"dimensions":[{"name":"acc","description":"x","weight":1.0}],"hard_gates":[]}',
        encoding="utf-8",
    )
    variants = root / "variants.json"
    variants.write_text(
        '{"system_prompts":["You are X."],"few_shot_examples":[]}',
        encoding="utf-8",
    )
    output = root / "artifact.json"
    return dataset, rubric, variants, output


def _invoke_calibrate_with_stub(
    monkeypatch,
    tmp_path: Path,
    artifact: CalibrationArtifact,
    *extra_args: str,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")
    from omegaprompt.commands import calibrate as calibrate_mod

    monkeypatch.setattr(calibrate_mod, "runtime_calibrate", lambda **_: artifact)
    dataset, rubric, variants, output = _write_minimal_cli_inputs(tmp_path)
    return runner.invoke(
        app,
        [
            "calibrate",
            str(dataset),
            "--rubric",
            str(rubric),
            "--variants",
            str(variants),
            "--output",
            str(output),
            *extra_args,
        ],
    )


def test_help_lists_subcommands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "calibrate" in result.stdout
    assert "report" in result.stdout
    assert "diff" in result.stdout
    assert "check-artifact" in result.stdout
    assert "gate" in result.stdout


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


def test_calibrate_help_documents_flags(monkeypatch):
    # Force a wide terminal so typer/rich's help renderer doesn't wrap
    # flag names across lines. Without this, a narrow CI runner shell
    # (default ~80 cols) splits "--rubric" into "--rub\nric" inside a
    # bordered table, and the substring assertions below fail. The test
    # is checking that documentation surfaces each flag, not how it's
    # paginated.
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("NO_COLOR", "1")
    result = runner.invoke(app, ["calibrate", "--help"])
    assert result.exit_code == 0
    # Strip any remaining ANSI escapes that survive NO_COLOR (typer's
    # help renderer still emits some bold codes) before substring search.
    import re

    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
    for flag in ("--rubric", "--variants", "--target-model", "--judge-model"):
        assert flag in plain, f"missing {flag} in help output"


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
        "ship_recommendation": "ship",
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
        "ship_recommendation": "ship",
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


def test_calibrate_exit_zero_for_ok_ship(monkeypatch, tmp_path: Path):
    result = _invoke_calibrate_with_stub(
        monkeypatch,
        tmp_path,
        _minimal_artifact(status="OK", ship=ShipRecommendation.SHIP),
    )

    assert result.exit_code == 0


def test_calibrate_exit_one_for_kc4_failure(monkeypatch, tmp_path: Path):
    result = _invoke_calibrate_with_stub(
        monkeypatch,
        tmp_path,
        _minimal_artifact(status="FAIL_KC4_GATE", ship=ShipRecommendation.HOLD),
    )

    assert result.exit_code == 1


def test_calibrate_exit_one_for_hard_gate_failure(monkeypatch, tmp_path: Path):
    result = _invoke_calibrate_with_stub(
        monkeypatch,
        tmp_path,
        _minimal_artifact(status="FAIL_HARD_GATES", ship=ShipRecommendation.BLOCK),
    )

    assert result.exit_code == 1


def test_calibrate_exit_one_for_ok_hold_in_ci_gate_mode(monkeypatch, tmp_path: Path):
    result = _invoke_calibrate_with_stub(
        monkeypatch,
        tmp_path,
        _minimal_artifact(status="OK", ship=ShipRecommendation.HOLD),
    )

    assert result.exit_code == 1


def test_calibrate_no_fail_on_gate_makes_hold_advisory(monkeypatch, tmp_path: Path):
    result = _invoke_calibrate_with_stub(
        monkeypatch,
        tmp_path,
        _minimal_artifact(status="OK", ship=ShipRecommendation.HOLD),
        "--no-fail-on-gate",
    )

    assert result.exit_code == 0


def test_calibrate_unknown_provider_exits_two(tmp_path: Path):
    dataset, rubric, variants, output = _write_minimal_cli_inputs(tmp_path)

    result = runner.invoke(
        app,
        [
            "calibrate",
            str(dataset),
            "--rubric",
            str(rubric),
            "--variants",
            str(variants),
            "--output",
            str(output),
            "--target-provider",
            "not-a-provider",
        ],
    )

    assert result.exit_code == 2
    assert "Unknown --target-provider" in result.output


def test_calibrate_missing_env_var_exits_two(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    dataset, rubric, variants, output = _write_minimal_cli_inputs(tmp_path)

    result = runner.invoke(
        app,
        [
            "calibrate",
            str(dataset),
            "--rubric",
            str(rubric),
            "--variants",
            str(variants),
            "--output",
            str(output),
            "--target-provider",
            "anthropic",
            "--judge-provider",
            "anthropic",
        ],
    )

    assert result.exit_code == 2
    assert "ANTHROPIC_API_KEY" in result.output


def test_report_invalid_artifact_exits_two(tmp_path: Path):
    artifact_path = tmp_path / "invalid.json"
    artifact_path.write_text('{"schema_version":"2.0"}', encoding="utf-8")

    result = runner.invoke(app, ["report", str(artifact_path)])

    assert result.exit_code == 2
    assert "INVALID_ARTIFACT" in result.output


def test_diff_invalid_artifact_exits_two(tmp_path: Path):
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    old_path.write_text('{"schema_version":"2.0"}', encoding="utf-8")
    new_path.write_text('{"schema_version":"2.0"}', encoding="utf-8")

    result = runner.invoke(app, ["diff", str(old_path), str(new_path)])

    assert result.exit_code == 2
    assert "INVALID_ARTIFACT" in result.output
