"""Tests for the overfit (train<->holdout transfer) metric surfacing."""

from __future__ import annotations

from pathlib import Path

from omegaprompt import extract_overfit_metrics, overfit_metrics_dict
from omegaprompt.core.artifact import load_artifact
from omegaprompt.core.overfit import (
    OVERFIT_SUMMARY_SCHEMA_VERSION,
    OverfitMetrics,
)
from omegaprompt.domain.result import CalibrationArtifact

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "examples" / "reference"


def _minimal_artifact(**overrides) -> CalibrationArtifact:
    base = dict(
        method="p1",
        unlock_k=1,
        best_params={"system_prompt_variant": 0},
        best_fitness=0.8,
        hard_gate_pass_rate=1.0,
        n_candidates_evaluated=1,
        total_api_calls=1,
    )
    base.update(overrides)
    return CalibrationArtifact(**base)


def test_extract_returns_overfit_metrics_model() -> None:
    a = load_artifact(REF / "reference_artifact.json")
    metrics = extract_overfit_metrics(a)
    assert isinstance(metrics, OverfitMetrics)
    assert metrics.schema_version == OVERFIT_SUMMARY_SCHEMA_VERSION


def test_clean_artifact_generalizes() -> None:
    a = load_artifact(REF / "reference_artifact.json")
    metrics = extract_overfit_metrics(a)
    assert metrics.available is True
    assert metrics.overfit_verdict == "GENERALIZES"
    assert metrics.walk_forward_passed is True


def test_kc4_failure_is_overfit_with_transfer_correlation() -> None:
    a = load_artifact(REF / "reference_fail_kc4_gate.json")
    metrics = extract_overfit_metrics(a)
    assert metrics.available is True
    assert metrics.overfit_verdict == "OVERFIT"
    assert metrics.transfer_correlation is not None
    assert metrics.transfer_correlation_status == "COMPUTED"
    assert metrics.walk_forward_passed is False


def test_no_walk_forward_is_unknown() -> None:
    a = _minimal_artifact()
    assert a.walk_forward is None
    metrics = extract_overfit_metrics(a)
    assert metrics.available is False
    assert metrics.overfit_verdict == "UNKNOWN"


def test_overfit_metrics_dict_is_json_ready() -> None:
    a = load_artifact(REF / "reference_artifact.json")
    d = overfit_metrics_dict(a)
    assert isinstance(d, dict)
    assert d["overfit_verdict"] == "GENERALIZES"
    assert d["schema_version"] == OVERFIT_SUMMARY_SCHEMA_VERSION


def test_extraction_does_not_mutate_artifact_schema() -> None:
    # The artifact must NOT gain an overfit field — schema stays "2.0" and
    # the serialized JSON is unchanged by extraction.
    a = load_artifact(REF / "reference_artifact.json")
    before = a.model_dump(mode="json")
    extract_overfit_metrics(a)
    after = a.model_dump(mode="json")
    assert before == after
    assert "overfit" not in after
    assert a.schema_version == "2.0"
