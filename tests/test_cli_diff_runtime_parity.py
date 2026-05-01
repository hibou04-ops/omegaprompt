"""Reviewer P0: CLI ``omegaprompt diff`` and ``runtime.diff`` agree.

Pre-fix the CLI had its own regression logic that didn't check
``new.status != \"OK\"`` or ``ship_recommendation in {BLOCK, HOLD}``.
A new artifact with ``status=FAIL_KC4_GATE`` but better metrics could
exit 0 from the CLI while ``runtime.diff()`` flagged it as a regression.

Now the CLI is a thin wrapper over ``runtime.diff()`` so the contract
is unified.
"""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from omegaprompt.cli import app
from omegaprompt.domain import ShipRecommendation
from omegaprompt.domain.result import CalibrationArtifact


runner = CliRunner()


def _write_artifact(path: Path, **overrides) -> None:
    base = {
        "method": "p1",
        "unlock_k": 3,
        "best_params": {"system_prompt_variant": 1},
        "best_fitness": 0.8,
        "calibrated_params": {"system_prompt_variant": 1},
        "calibrated_fitness": 0.8,
        "neutral_fitness": 0.5,
        "hard_gate_pass_rate": 1.0,
        "quality_per_cost_best": 0.2,
        "quality_per_latency_best": 0.1,
        "n_candidates_evaluated": 1,
        "total_api_calls": 1,
        "ship_recommendation": ShipRecommendation.SHIP,
        "status": "OK",
    }
    base.update(overrides)
    artifact = CalibrationArtifact(**base)
    path.write_text(artifact.model_dump_json(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Status-only regression: CLI matches runtime.
# ---------------------------------------------------------------------------


def test_cli_diff_fails_when_candidate_status_not_ok(tmp_path: Path):
    """Pre-fix scenario: new artifact has FAIL_KC4_GATE but better
    metrics. CLI used to treat this as OK because metrics improved."""
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    _write_artifact(old, calibrated_fitness=0.7, best_fitness=0.7)
    _write_artifact(new,
        calibrated_fitness=0.95,  # metrics improved...
        best_fitness=0.95,
        status="FAIL_KC4_GATE",  # ...but candidate failed its gate
    )
    result = runner.invoke(app, ["diff", str(old), str(new)])
    assert result.exit_code == 1, (
        "CLI must surface candidate's non-OK status as regression, "
        "matching runtime.diff() behaviour."
    )


def test_cli_diff_fails_when_candidate_ship_recommendation_is_block(tmp_path: Path):
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    _write_artifact(old, ship_recommendation=ShipRecommendation.SHIP)
    _write_artifact(new,
        calibrated_fitness=0.95,
        best_fitness=0.95,
        ship_recommendation=ShipRecommendation.BLOCK,
    )
    result = runner.invoke(app, ["diff", str(old), str(new)])
    assert result.exit_code == 1


def test_cli_diff_fails_when_candidate_ship_recommendation_is_hold(tmp_path: Path):
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    _write_artifact(old, ship_recommendation=ShipRecommendation.SHIP)
    _write_artifact(new,
        calibrated_fitness=0.95,
        best_fitness=0.95,
        ship_recommendation=ShipRecommendation.HOLD,
    )
    result = runner.invoke(app, ["diff", str(old), str(new)])
    assert result.exit_code == 1


def test_cli_diff_passes_when_candidate_clean(tmp_path: Path):
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    _write_artifact(old, calibrated_fitness=0.7, best_fitness=0.7)
    _write_artifact(new, calibrated_fitness=0.85, best_fitness=0.85)
    result = runner.invoke(app, ["diff", str(old), str(new)])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# --no-fail-on-regression keeps the CLI advisory.
# ---------------------------------------------------------------------------


def test_cli_diff_no_fail_on_regression_returns_zero_even_on_status_fail(tmp_path: Path):
    """Some workflows want to compute the diff without failing the
    build — surfaces the regression but exits 0."""
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    _write_artifact(old)
    _write_artifact(new, status="FAIL_KC4_GATE")
    result = runner.invoke(app, [
        "diff", str(old), str(new), "--no-fail-on-regression",
    ])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Markdown output mirrors runtime.diff() format.
# ---------------------------------------------------------------------------


def test_cli_diff_outputs_markdown_with_diff_header(tmp_path: Path):
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    _write_artifact(old)
    _write_artifact(new, calibrated_fitness=0.85, best_fitness=0.85)
    result = runner.invoke(app, ["diff", str(old), str(new)])
    assert "# omegaprompt diff" in result.stdout
