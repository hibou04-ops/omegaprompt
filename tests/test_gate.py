"""Tests for the ship gate (core.gate) and the ``omegaprompt gate`` CLI."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from omegaprompt import run_gate
from omegaprompt.cli import app
from omegaprompt.core.gate import GATE_SCHEMA_VERSION, GateResult, run_gate as core_run_gate

runner = CliRunner()

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "examples" / "reference"
CLEAN = REF / "reference_artifact.json"
FAIL_KC4 = REF / "reference_fail_kc4_gate.json"
FAIL_HARD = REF / "reference_fail_hard_gates.json"
DEGRADED = REF / "reference_provider_degradation.json"


def test_gate_passes_clean_artifact() -> None:
    result = run_gate(CLEAN)
    assert isinstance(result, GateResult)
    assert result.passed is True
    assert result.exit_code == 0
    assert result.integrity_valid is True
    assert result.release_approved is True
    assert result.overfit_verdict == "GENERALIZES"
    assert result.blocking_reasons == []
    assert result.gate_schema_version == GATE_SCHEMA_VERSION


def test_gate_blocks_kc4_failure() -> None:
    result = run_gate(FAIL_KC4)
    assert result.passed is False
    assert result.exit_code == 1
    assert result.release_approved is False
    assert result.overfit_verdict == "OVERFIT"
    assert any("release-approved" in r for r in result.blocking_reasons)
    assert any("overfit" in r for r in result.blocking_reasons)


def test_gate_blocks_hard_gate_failure() -> None:
    result = run_gate(FAIL_HARD)
    assert result.passed is False
    assert result.exit_code == 1
    assert result.blocking_reasons


def test_gate_missing_file_exits_two() -> None:
    result = run_gate(REF / "does_not_exist.json")
    assert result.passed is False
    assert result.exit_code == 2
    assert result.integrity_valid is False


def test_gate_require_generalization_toggle_does_not_unblock_failed_release() -> None:
    # Disabling generalization must still block on the non-ship status.
    result = run_gate(FAIL_KC4, require_generalization=False)
    assert result.passed is False
    # The generalization-specific blocking reason should be gone, but the
    # release-approval block remains.
    assert not any("overfit" in r for r in result.blocking_reasons)
    assert any("release-approved" in r for r in result.blocking_reasons)


def test_top_level_run_gate_is_core_run_gate() -> None:
    assert run_gate is core_run_gate


# --------------------------- CLI -----------------------------


def test_cli_gate_clean_exit_zero() -> None:
    result = runner.invoke(app, ["gate", str(CLEAN)])
    assert result.exit_code == 0
    assert "PASSED: True" in result.stdout


def test_cli_gate_fail_exit_one() -> None:
    result = runner.invoke(app, ["gate", str(FAIL_KC4)])
    assert result.exit_code == 1


def test_cli_gate_json_format_is_machine_readable() -> None:
    result = runner.invoke(app, ["gate", str(CLEAN), "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert payload["gate_schema_version"] == GATE_SCHEMA_VERSION
    assert payload["overfit"]["overfit_verdict"] == "GENERALIZES"


def test_cli_gate_json_format_failure_exit_one_and_parseable() -> None:
    result = runner.invoke(app, ["gate", str(FAIL_KC4), "--format", "json"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["passed"] is False
    assert payload["blocking_reasons"]


def test_cli_gate_invalid_format_exits_two() -> None:
    result = runner.invoke(app, ["gate", str(CLEAN), "--format", "xml"])
    assert result.exit_code == 2


def test_cli_gate_missing_file_exits_two() -> None:
    result = runner.invoke(app, ["gate", str(REF / "nope.json")])
    assert result.exit_code == 2


def test_gate_json_is_byte_stable() -> None:
    a = run_gate(CLEAN).to_json_dict()
    b = run_gate(CLEAN).to_json_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
