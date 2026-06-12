"""Determinism / replay hardening for the 2.1.0 JSON surfaces.

The gate JSON and report-summary JSON are CI-consumable and must be
byte-stable: re-running over the same golden artifact, and round-tripping
the artifact through save/load, must produce identical bytes. This extends
the offline golden-reference replay so CI catches any nondeterminism in
the new surfaces without depending on the network.
"""

from __future__ import annotations

import json
from pathlib import Path

from omegaprompt.core.artifact import load_artifact, save_artifact
from omegaprompt.core.gate import run_gate
from omegaprompt.reporting import render_summary_json
from tests.helpers import workspace_tmpdir

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "examples" / "reference"

GOLDEN = [
    "reference_artifact.json",
    "reference_fail_kc4_gate.json",
    "reference_fail_hard_gates.json",
    "reference_provider_degradation.json",
    "reference_diff_regression.json",
]


def test_gate_json_is_byte_stable_across_repeats() -> None:
    for name in GOLDEN:
        path = REF / name
        first = json.dumps(run_gate(path).to_json_dict(), sort_keys=True)
        second = json.dumps(run_gate(path).to_json_dict(), sort_keys=True)
        assert first == second, name


def test_report_summary_json_is_byte_stable_across_repeats() -> None:
    for name in GOLDEN:
        a = load_artifact(REF / name)
        assert render_summary_json(a) == render_summary_json(a), name


def test_summary_json_survives_save_load_roundtrip() -> None:
    # Round-tripping the artifact through disk must not change the summary
    # bytes — proves the summary depends only on stable artifact fields.
    with workspace_tmpdir() as tmp:
        for name in GOLDEN:
            original = load_artifact(REF / name)
            before = render_summary_json(original)

            out = tmp / name
            save_artifact(original, out)
            reloaded = load_artifact(out)
            after = render_summary_json(reloaded)
            assert before == after, name


def test_gate_verdict_is_stable_save_load_roundtrip() -> None:
    with workspace_tmpdir() as tmp:
        for name in GOLDEN:
            original = load_artifact(REF / name)
            ref_result = run_gate(REF / name)

            out = tmp / name
            save_artifact(original, out)
            round_result = run_gate(out)

            assert round_result.passed == ref_result.passed, name
            assert round_result.exit_code == ref_result.exit_code, name
            assert round_result.overfit_verdict == ref_result.overfit_verdict, name


def test_golden_artifact_schema_unchanged() -> None:
    # 2.1.0 must not have mutated the artifact schema — every golden file is
    # still schema "2.0" and loads cleanly.
    for name in GOLDEN:
        a = load_artifact(REF / name)
        assert a.schema_version == "2.0", name
