from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

from tools import reproduce_golden_reference
from omegaprompt.core.artifact import load_artifact
from omegaprompt.core.artifact_integrity import (
    check_artifact_integrity,
    normalized_artifact_hash,
)
from omegaprompt.runtime import diff as artifact_diff

ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "examples" / "reference" / "reproduce_reference_artifact.py"
MANIFEST_PATH = ROOT / "examples" / "reference" / "golden_manifest.json"


def _load_harness():
    spec = importlib.util.spec_from_file_location("test_golden_harness", HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _classification(report) -> str:
    if not report.schema_valid:
        return "schema_error"
    if not report.valid:
        return "integrity_error"
    if report.release_approved:
        return "release_approved"
    return "valid_non_release"


def test_golden_manifest_covers_required_cases_and_modern_walk_forward_fields() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in manifest["cases"]}

    assert {
        "clean_ok_ship",
        "fail_kc4_gate",
        "fail_hard_gates",
        "provider_degradation",
        "diff_regression_candidate",
    } <= set(cases)
    for case in cases.values():
        assert case["reproducible_command"] == "python examples/reference/reproduce_reference_artifact.py"
        assert isinstance(case["exact_metrics_may_be_displayed"], bool)
        artifact = json.loads((ROOT / case["artifact"]).read_text(encoding="utf-8"))
        walk_forward = artifact["walk_forward"]
        assert {
            "validation_mode",
            "shared_item_count",
            "kc4_status",
            "max_gap_threshold",
            "min_kc4_threshold",
            "passed",
        } <= set(walk_forward)


def test_reproduced_artifacts_match_manifest_hashes_and_integrity() -> None:
    harness = _load_harness()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in manifest["cases"]}

    with contextlib.redirect_stdout(io.StringIO()):
        generated = harness.build_golden_artifacts()

    for case_id, artifact in generated.items():
        case = cases[case_id]
        disk_artifact = load_artifact(ROOT / case["artifact"])
        report = check_artifact_integrity(ROOT / case["artifact"])

        assert normalized_artifact_hash(artifact) == case["normalized_artifact_hash"]
        assert normalized_artifact_hash(disk_artifact) == case["normalized_artifact_hash"]
        assert _classification(report) == case["expected_integrity_classification"]
        assert str(disk_artifact.status.value) == case["expected_status"]
        assert disk_artifact.ship_recommendation.value == case["expected_ship_recommendation"]
        assert disk_artifact.walk_forward is not None
        assert disk_artifact.walk_forward.validation_mode == case["expected_validation_mode"]


def test_manifest_diff_regression_case_is_a_real_runtime_regression() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in manifest["cases"]}
    case = cases["diff_regression_candidate"]
    baseline = cases[case["diff_baseline_case_id"]]

    result = artifact_diff(ROOT / baseline["artifact"], ROOT / case["artifact"])

    assert result.regressed is True
    assert case["expected_diff_regression"] is True
    assert result.regression_reasons


def test_reproduce_golden_reference_check_command_passes() -> None:
    result = subprocess.run(
        [sys.executable, "tools/reproduce_golden_reference.py", "--check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Golden reference artifacts are fresh." in result.stdout


def test_reproduce_golden_reference_detects_stale_manifest_hash(monkeypatch, tmp_path: Path) -> None:
    stale_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    stale_manifest["cases"][0]["normalized_artifact_hash"] = "0" * 64
    manifest_path = tmp_path / "golden_manifest.json"
    manifest_path.write_text(json.dumps(stale_manifest), encoding="utf-8")
    monkeypatch.setattr(reproduce_golden_reference, "MANIFEST_PATH", manifest_path)

    harness = _load_harness()
    errors = reproduce_golden_reference._check(harness)

    assert any("manifest normalized_artifact_hash" in error for error in errors)
