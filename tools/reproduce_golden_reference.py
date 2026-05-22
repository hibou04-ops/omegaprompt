#!/usr/bin/env python
"""Regenerate or check the deterministic offline golden reference harness."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "examples" / "reference" / "reproduce_reference_artifact.py"
MANIFEST_PATH = ROOT / "examples" / "reference" / "golden_manifest.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check generated artifacts and manifest without modifying files.",
    )
    args = parser.parse_args(argv)

    try:
        harness = _load_harness()
    except ImportError as exc:
        print(f"TOOLING_MISSING: {exc}", file=sys.stderr)
        return 2

    if not args.check:
        harness.write_golden_artifacts()
        print("Wrote examples/reference golden artifacts and manifest.")
        return 0

    errors = _check(harness)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 2 if any(error.startswith("ENVIRONMENT_BLOCKED") for error in errors) else 1
    print("Golden reference artifacts are fresh.")
    return 0


def _load_harness():
    spec = importlib.util.spec_from_file_location("golden_reference_harness", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load harness module at {HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check(harness) -> list[str]:
    from omegaprompt.core.artifact import load_artifact
    from omegaprompt.core.artifact_integrity import (
        canonical_artifact_json,
        check_artifact_integrity,
        normalized_artifact_hash,
    )
    from omegaprompt.runtime import diff as artifact_diff

    errors: list[str] = []
    if not MANIFEST_PATH.exists():
        return [f"ENVIRONMENT_BLOCKED: missing {MANIFEST_PATH}"]

    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except OSError as exc:
        return [f"ENVIRONMENT_BLOCKED: cannot read {MANIFEST_PATH}: {exc}"]
    except json.JSONDecodeError as exc:
        return [f"GOLDEN_DRIFT: manifest is not valid JSON: {exc}"]

    manifest_cases = {
        case.get("case_id"): case for case in manifest.get("cases", [])
        if isinstance(case, dict)
    }
    with contextlib.redirect_stdout(io.StringIO()):
        expected_artifacts = harness.build_golden_artifacts()

    if set(manifest_cases) != set(expected_artifacts):
        errors.append(
            "GOLDEN_DRIFT: manifest case ids differ from generated harness: "
            f"manifest={sorted(manifest_cases)} generated={sorted(expected_artifacts)}"
        )

    for case_id, expected_artifact in expected_artifacts.items():
        manifest_case = manifest_cases.get(case_id)
        if manifest_case is None:
            continue

        artifact_path = ROOT / "examples" / "reference" / harness.CASE_FILES[case_id]
        if not artifact_path.exists():
            errors.append(f"ENVIRONMENT_BLOCKED: missing {artifact_path}")
            continue

        try:
            disk_artifact = load_artifact(artifact_path)
        except Exception as exc:
            errors.append(f"GOLDEN_DRIFT: {artifact_path} failed schema validation: {exc}")
            continue

        expected_hash = normalized_artifact_hash(expected_artifact)
        disk_hash = normalized_artifact_hash(disk_artifact)
        if disk_hash != expected_hash:
            errors.append(
                f"GOLDEN_DRIFT: {case_id} normalized artifact hash changed: "
                f"disk={disk_hash} expected={expected_hash}"
            )
        if canonical_artifact_json(disk_artifact) != canonical_artifact_json(expected_artifact):
            errors.append(f"GOLDEN_DRIFT: {case_id} normalized artifact JSON is stale")

        report = check_artifact_integrity(artifact_path)
        classification = _integrity_classification(report)
        expected_validation_mode = (
            expected_artifact.walk_forward.validation_mode
            if expected_artifact.walk_forward is not None
            else None
        )
        expected_fields = {
            "expected_status": _enum_value(expected_artifact.status),
            "expected_ship_recommendation": _enum_value(expected_artifact.ship_recommendation),
            "expected_validation_mode": expected_validation_mode,
            "expected_integrity_classification": classification,
            "normalized_artifact_hash": expected_hash,
        }
        for key, expected_value in expected_fields.items():
            actual_value = manifest_case.get(key)
            if actual_value != expected_value:
                errors.append(
                    f"GOLDEN_DRIFT: {case_id} manifest {key}={actual_value!r}, "
                    f"expected {expected_value!r}"
                )

        if not _walk_forward_has_modern_fields(artifact_path):
            errors.append(f"GOLDEN_DRIFT: {case_id} missing modern walk_forward fields")

        expected_diff = manifest_case.get("expected_diff_regression")
        baseline_case = manifest_case.get("diff_baseline_case_id")
        if expected_diff is not None:
            baseline_path = ROOT / "examples" / "reference" / harness.CASE_FILES[str(baseline_case)]
            diff_result = artifact_diff(baseline_path, artifact_path)
            if diff_result.regressed != expected_diff:
                errors.append(
                    f"GOLDEN_DRIFT: {case_id} diff regression={diff_result.regressed}, "
                    f"expected {expected_diff}"
                )

    return errors


def _walk_forward_has_modern_fields(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    walk_forward = data.get("walk_forward")
    if not isinstance(walk_forward, dict):
        return False
    required = {
        "validation_mode",
        "shared_item_count",
        "kc4_status",
        "max_gap_threshold",
        "min_kc4_threshold",
        "passed",
    }
    return required <= set(walk_forward)


def _integrity_classification(report) -> str:
    if not report.schema_valid:
        return "schema_error"
    if not report.valid:
        return "integrity_error"
    if report.release_approved:
        return "release_approved"
    return "valid_non_release"


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


if __name__ == "__main__":
    raise SystemExit(main())
