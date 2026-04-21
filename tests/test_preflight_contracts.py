"""Preflight contract tests - shape, enums, severity ordering."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omegaprompt.preflight.contracts import (
    AnalyticalFinding,
    EndpointMeasurement,
    JudgeQualityMeasurement,
    PerformanceMeasurement,
    PreflightReport,
    PreflightSeverity,
    PreflightStatus,
)


def test_preflight_severity_ordering():
    severity_rank = {
        PreflightSeverity.LOW: 0,
        PreflightSeverity.MEDIUM: 1,
        PreflightSeverity.HIGH: 2,
        PreflightSeverity.BLOCKER: 3,
    }
    assert severity_rank[PreflightSeverity.BLOCKER] > severity_rank[PreflightSeverity.HIGH]
    assert severity_rank[PreflightSeverity.HIGH] > severity_rank[PreflightSeverity.MEDIUM]
    assert severity_rank[PreflightSeverity.MEDIUM] > severity_rank[PreflightSeverity.LOW]


def test_preflight_status_values():
    assert PreflightStatus.PROCEED.value == "proceed"
    assert PreflightStatus.ADAPT.value == "adapt"
    assert PreflightStatus.ABORT.value == "abort"


def test_analytical_finding_requires_label_and_trap_id():
    with pytest.raises(ValidationError):
        AnalyticalFinding(label="REAL", hypothesis="x")
    with pytest.raises(ValidationError):
        AnalyticalFinding(trap_id="t1", hypothesis="x")


def test_analytical_finding_happy_path():
    f = AnalyticalFinding(
        trap_id="self_agreement_bias",
        label="REAL",
        hypothesis="Target and judge share vendor",
        severity=PreflightSeverity.HIGH,
        note="openai/gpt-4o + openai/gpt-4o",
        remediation="Use cross-vendor judge",
        cite="calibration.yaml:12",
    )
    assert f.severity == PreflightSeverity.HIGH
    assert "share vendor" in f.hypothesis


def test_analytical_finding_forbids_extras():
    with pytest.raises(ValidationError):
        AnalyticalFinding(
            trap_id="x",
            label="REAL",
            hypothesis="y",
            unknown_field="bad",
        )


def test_judge_quality_clamps():
    with pytest.raises(ValidationError):
        JudgeQualityMeasurement(consistency=1.5)
    with pytest.raises(ValidationError):
        JudgeQualityMeasurement(consistency=-0.1)
    m = JudgeQualityMeasurement(consistency=0.85, anchoring_usage=0.6, scale_monotonic=True, samples=3)
    assert m.consistency == 0.85


def test_endpoint_measurement_reliability_bounds():
    with pytest.raises(ValidationError):
        EndpointMeasurement(schema_reliability=1.1)
    m = EndpointMeasurement(
        schema_reliability=0.67,
        context_budget_margin=0.3,
        caching_active=False,
        silent_degradation_detected=True,
    )
    assert m.silent_degradation_detected is True


def test_performance_measurement_defaults():
    m = PerformanceMeasurement()
    assert m.noise_floor == 0.0
    assert m.mean_call_latency_ms == 0.0


def test_preflight_report_worst_severity_with_no_findings():
    r = PreflightReport()
    assert r.worst_severity() == PreflightSeverity.LOW
    assert not r.any_real_or_new()


def test_preflight_report_worst_severity_picks_highest():
    r = PreflightReport(
        analytical_findings=[
            AnalyticalFinding(trap_id="a", label="GHOST", hypothesis="x", severity=PreflightSeverity.LOW),
            AnalyticalFinding(trap_id="b", label="REAL", hypothesis="y", severity=PreflightSeverity.HIGH),
            AnalyticalFinding(trap_id="c", label="NEW", hypothesis="z", severity=PreflightSeverity.MEDIUM),
        ]
    )
    assert r.worst_severity() == PreflightSeverity.HIGH
    assert r.any_real_or_new() is True


def test_preflight_report_any_real_or_new_false_on_ghosts_only():
    r = PreflightReport(
        analytical_findings=[
            AnalyticalFinding(trap_id="a", label="GHOST", hypothesis="x"),
            AnalyticalFinding(trap_id="b", label="UNRESOLVED", hypothesis="y"),
        ]
    )
    assert r.any_real_or_new() is False
