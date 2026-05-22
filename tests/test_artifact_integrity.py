from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from omegaprompt.cli import app
from omegaprompt.core.artifact_integrity import check_artifact_integrity
from tests.helpers import workspace_tmpdir

runner = CliRunner()


def _capabilities(provider: str = "anthropic", *, ship_grade: bool = True) -> dict:
    return {
        "provider": provider,
        "tier": "tier_2_cloud_grade",
        "supports_strict_schema": True,
        "supports_json_object": True,
        "supports_reasoning_profiles": True,
        "supports_usage_accounting": True,
        "supports_llm_judge": True,
        "ship_grade_judge": ship_grade,
        "supports_tools": False,
        "experimental": False,
        "placeholder": False,
        "notes": ["test fixture"],
    }


def _walk_forward(**overrides: object) -> dict:
    data = {
        "train_best_fitness": 0.9,
        "test_fitness": 0.85,
        "generalization_gap": 0.0555555555555556,
        "gap_status": "OK",
        "validation_mode": "auto",
        "shared_item_count": 3,
        "kc4_correlation": 0.8,
        "kc4_status": "COMPUTED",
        "max_gap_threshold": 0.25,
        "min_kc4_threshold": 0.5,
        "passed": True,
    }
    data.update(overrides)
    return data


def _artifact(**overrides: object) -> dict:
    data = {
        "schema_version": "2.0",
        "engine_name": "omegaprompt",
        "method": "p1",
        "unlock_k": 2,
        "selected_profile": "guarded",
        "neutral_baseline_params": {"system_prompt_variant": 0},
        "calibrated_params": {"system_prompt_variant": 1},
        "neutral_fitness": 0.7,
        "calibrated_fitness": 0.9,
        "uplift_absolute": 0.2,
        "uplift_percent": 28.5714285714,
        "quality_per_cost_neutral": 0.1,
        "quality_per_cost_best": 0.2,
        "quality_per_latency_neutral": 0.01,
        "quality_per_latency_best": 0.02,
        "boundary_warnings": [],
        "degraded_capabilities": [],
        "ship_recommendation": "ship",
        "stayed_within_guarded_boundaries": True,
        "additional_uplift_from_boundary_crossing": 0.0,
        "relaxed_safeguards": [],
        "guarded_boundary_crossed": False,
        "cost_basis": "normalized_token_units",
        "best_params": {"system_prompt_variant": 1},
        "best_fitness": 0.9,
        "walk_forward": _walk_forward(),
        "hard_gate_pass_rate": 1.0,
        "sensitivity_ranking": [
            {"axis": "system_prompt_variant", "gini_delta": 0.2, "rank": 0}
        ],
        "n_candidates_evaluated": 2,
        "total_api_calls": 4,
        "usage_summary": {"input_tokens": 10, "output_tokens": 5},
        "latency_summary_ms": {},
        "target_provider": "anthropic",
        "target_model": "fixture-target",
        "judge_provider": "anthropic",
        "judge_model": "fixture-judge",
        "target_capabilities": _capabilities("anthropic"),
        "judge_capabilities": _capabilities("anthropic"),
        "status": "OK",
        "rationale": "fixture passed",
        "adaptation_summary": None,
    }
    data.update(overrides)
    return data


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _ids(report) -> set[str]:
    return {finding.id for finding in report.findings}


def test_current_reference_artifact_is_valid_and_release_approved() -> None:
    report = check_artifact_integrity("examples/reference/reference_artifact.json")

    assert report.schema_valid is True
    assert report.valid is True
    assert report.release_approved is True
    assert report.canonical_sha256
    assert report.strict_blocking_findings == 0


def test_cli_check_artifact_outputs_machine_readable_json_for_reference() -> None:
    result = runner.invoke(
        app,
        [
            "check-artifact",
            "examples/reference/reference_artifact.json",
            "--strict",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["release_approved"] is True
    assert payload["schema_version"] == "2.0"


def test_strict_cli_fails_for_status_ship_mismatch() -> None:
    data = _artifact(
        status="FAIL_KC4_GATE",
        ship_recommendation="ship",
        walk_forward=_walk_forward(
            test_fitness=0.4,
            generalization_gap=0.5555555555555556,
            passed=False,
        ),
        rationale="KC4 failed",
    )
    with workspace_tmpdir() as tmp:
        path = _write(tmp / "bad.json", data)

        result = runner.invoke(app, ["check-artifact", str(path), "--strict", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    ids = {finding["id"] for finding in payload["findings"]}
    assert "FAIL_KC4_GATE_CANNOT_SHIP" in ids
    assert payload["release_approved"] is False


def test_legacy_walk_forward_shape_is_not_silently_accepted() -> None:
    legacy_wf = {
        "train_best_fitness": 0.9,
        "test_fitness": 0.85,
        "generalization_gap": 0.0555555555555556,
        "kc4_correlation": 0.8,
        "passed": True,
    }
    with workspace_tmpdir() as tmp:
        path = _write(tmp / "legacy.json", _artifact(walk_forward=legacy_wf))

        report = check_artifact_integrity(path)

    assert report.valid is False
    assert "LEGACY_WALK_FORWARD_SHAPE" in _ids(report)


def test_missing_provider_capabilities_are_classified() -> None:
    with workspace_tmpdir() as tmp:
        path = _write(tmp / "missing-caps.json", _artifact(target_capabilities=None))

        report = check_artifact_integrity(path)

    assert report.valid is False
    assert "MISSING_TARGET_CAPABILITIES" in _ids(report)


def test_invalid_kc4_computed_state_uses_existing_pydantic_validation() -> None:
    bad_wf = _walk_forward(kc4_status="COMPUTED", kc4_correlation=None)
    with workspace_tmpdir() as tmp:
        path = _write(tmp / "bad-kc4.json", _artifact(walk_forward=bad_wf))

        report = check_artifact_integrity(path)

    assert report.schema_valid is False
    assert "ARTIFACT_SCHEMA_VALIDATION_FAILED" in _ids(report)


def test_unsupported_schema_version_is_blocking() -> None:
    with workspace_tmpdir() as tmp:
        path = _write(tmp / "unsupported-schema.json", _artifact(schema_version="3.0"))

        report = check_artifact_integrity(path)

    assert report.schema_valid is False
    assert report.valid is False
    assert "SCHEMA_VERSION_UNSUPPORTED" in _ids(report)


def test_unknown_extra_fields_are_visible_but_do_not_relax_validation() -> None:
    with workspace_tmpdir() as tmp:
        path = _write(tmp / "extra.json", _artifact(unexpected_public_claim="hidden"))

        report = check_artifact_integrity(path)

    assert report.valid is True
    assert "UNKNOWN_EXTRA_FIELDS_VISIBLE" in _ids(report)
    assert report.counts["warnings"] == 1


def test_degraded_capabilities_and_relaxed_safeguards_are_visible() -> None:
    degraded = {
        "capability": "structured_output",
        "requested": "strict_schema",
        "applied": "json_object_parse",
        "reason": "fixture fallback",
        "user_visible_note": "fixture note",
        "affects_guarded_boundary": False,
    }
    relaxed = {
        "name": "ship_grade_judge",
        "reason": "expedition fixture",
        "increased_risk": "judge not release approved",
    }
    data = _artifact(
        selected_profile="expedition",
        ship_recommendation="experiment",
        stayed_within_guarded_boundaries=False,
        guarded_boundary_crossed=True,
        degraded_capabilities=[degraded],
        relaxed_safeguards=[relaxed],
        rationale="expedition boundary crossing recorded",
    )
    with workspace_tmpdir() as tmp:
        path = _write(tmp / "visible.json", data)

        report = check_artifact_integrity(path)

    assert report.valid is True
    assert report.release_approved is False
    assert "DEGRADED_CAPABILITIES_VISIBLE" in _ids(report)
    assert "RELAXED_SAFEGUARDS_VISIBLE" in _ids(report)
    assert report.degraded_capabilities == [degraded]
    assert report.relaxed_safeguards == [relaxed]


def test_guarded_boundary_degradation_cannot_be_hidden() -> None:
    degraded = {
        "capability": "strict_schema",
        "requested": "strict_schema",
        "applied": "json_object_parse",
        "reason": "fixture fallback",
        "user_visible_note": "fixture note",
        "affects_guarded_boundary": True,
    }
    data = _artifact(degraded_capabilities=[degraded])
    with workspace_tmpdir() as tmp:
        path = _write(tmp / "hidden-boundary.json", data)

        report = check_artifact_integrity(path)

    assert report.valid is False
    assert "GUARDED_DEGRADATION_HIDDEN" in _ids(report)
    assert "GUARDED_DEGRADATION_CANNOT_SHIP" in _ids(report)


def test_ok_hold_requires_explicit_explanation() -> None:
    data = _artifact(
        ship_recommendation="hold",
        rationale="",
        boundary_warnings=[],
        degraded_capabilities=[],
        relaxed_safeguards=[],
        adaptation_summary=None,
    )
    with workspace_tmpdir() as tmp:
        path = _write(tmp / "unexplained-hold.json", data)

        report = check_artifact_integrity(path)

    assert report.valid is False
    assert report.release_approved is False
    assert "OK_HOLD_BLOCK_REQUIRES_EXPLANATION" in _ids(report)


def test_malformed_json_is_reported_as_schema_error() -> None:
    with workspace_tmpdir() as tmp:
        path = tmp / "malformed.json"
        path.write_text("{not json", encoding="utf-8")

        report = check_artifact_integrity(path)

    assert report.schema_valid is False
    assert "ARTIFACT_JSON_INVALID" in _ids(report)


def test_missing_file_is_environment_blocked() -> None:
    with workspace_tmpdir() as tmp:
        report = check_artifact_integrity(tmp / "missing.json")

    assert report.environment_blocked is True
    assert "ARTIFACT_FILE_NOT_FOUND" in _ids(report)
