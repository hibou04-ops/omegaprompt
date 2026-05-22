"""Zero-network semantic integrity checks for CalibrationArtifact JSON files."""

from __future__ import annotations

import hashlib
import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, computed_field

from omegaprompt.core.artifact import load_artifact
from omegaprompt.domain.profiles import ExecutionProfile, ShipRecommendation
from omegaprompt.domain.result import ArtifactStatus, CalibrationArtifact

SUPPORTED_SCHEMA_VERSIONS = {"2.0"}

Severity = Literal["INFO", "WARNING", "ERROR"]


class IntegrityFinding(BaseModel):
    """One machine-readable integrity finding."""

    model_config = ConfigDict(extra="forbid")

    id: str
    severity: Severity
    category: str
    message: str
    path: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ArtifactIntegrityReport(BaseModel):
    """CI-friendly result for ``omegaprompt check-artifact``."""

    model_config = ConfigDict(extra="forbid")

    artifact_path: str
    schema_valid: bool
    semantic_valid: bool
    valid: bool
    release_approved: bool
    schema_version: str | None = None
    status: str | None = None
    ship_recommendation: str | None = None
    selected_profile: str | None = None
    canonical_sha256: str | None = None
    degraded_capabilities: list[dict[str, Any]] = Field(default_factory=list)
    relaxed_safeguards: list[dict[str, Any]] = Field(default_factory=list)
    guarded_boundary_crossed: bool | None = None
    stayed_within_guarded_boundaries: bool | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    findings: list[IntegrityFinding] = Field(default_factory=list)

    @computed_field(return_type=int)
    @property
    def strict_blocking_findings(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "ERROR")

    @computed_field(return_type=bool)
    @property
    def environment_blocked(self) -> bool:
        return any(finding.category == "ENVIRONMENT_BLOCKED" for finding in self.findings)


def check_artifact_integrity(path: str | Path) -> ArtifactIntegrityReport:
    """Load a CalibrationArtifact through the existing schema path, then audit it.

    The checker does not call providers, does not fetch network state, and does
    not relax Pydantic validation. Raw JSON is inspected only to make legacy
    fields and unknown extras visible after the model has normalized them.
    """

    artifact_path = Path(path)
    findings: list[IntegrityFinding] = []

    try:
        raw_text = artifact_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        findings.append(
            _finding(
                "ARTIFACT_FILE_NOT_FOUND",
                "ERROR",
                "ENVIRONMENT_BLOCKED",
                f"Artifact file is not accessible: {artifact_path}",
            )
        )
        return _report(path=artifact_path, findings=findings, schema_valid=False)
    except OSError as exc:
        findings.append(
            _finding(
                "ARTIFACT_FILE_INACCESSIBLE",
                "ERROR",
                "ENVIRONMENT_BLOCKED",
                f"Artifact file could not be read: {exc}",
            )
        )
        return _report(path=artifact_path, findings=findings, schema_valid=False)

    try:
        raw = json.loads(raw_text)
    except JSONDecodeError as exc:
        findings.append(
            _finding(
                "ARTIFACT_JSON_INVALID",
                "ERROR",
                "schema",
                f"Artifact is not valid JSON: {exc.msg}",
                details={"line": exc.lineno, "column": exc.colno},
            )
        )
        return _report(path=artifact_path, findings=findings, schema_valid=False)

    if not isinstance(raw, dict):
        findings.append(
            _finding(
                "ARTIFACT_JSON_NOT_OBJECT",
                "ERROR",
                "schema",
                "CalibrationArtifact JSON must be an object.",
            )
        )
        return _report(path=artifact_path, findings=findings, schema_valid=False)

    raw_schema_version = raw.get("schema_version")
    if raw_schema_version is None:
        findings.append(
            _finding(
                "SCHEMA_VERSION_MISSING",
                "ERROR",
                "schema",
                "schema_version is missing instead of explicitly declaring a supported artifact schema.",
                path="schema_version",
            )
        )
    elif raw_schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        findings.append(
            _finding(
                "SCHEMA_VERSION_UNSUPPORTED",
                "ERROR",
                "schema",
                f"schema_version={raw_schema_version!r} is not supported by this checker.",
                path="schema_version",
                details={"supported": sorted(SUPPORTED_SCHEMA_VERSIONS)},
            )
        )

    try:
        artifact = load_artifact(artifact_path)
    except ValidationError as exc:
        findings.append(
            _finding(
                "ARTIFACT_SCHEMA_VALIDATION_FAILED",
                "ERROR",
                "schema",
                "Existing CalibrationArtifact Pydantic validation rejected the artifact.",
                details={"errors": exc.errors(include_url=False)},
            )
        )
        return _report(
            path=artifact_path,
            findings=findings,
            schema_valid=False,
            schema_version=_string_or_none(raw_schema_version),
        )
    except Exception as exc:
        findings.append(
            _finding(
                "ARTIFACT_LOAD_FAILED",
                "ERROR",
                "schema",
                f"Existing CalibrationArtifact loader rejected the artifact: {exc}",
            )
        )
        return _report(
            path=artifact_path,
            findings=findings,
            schema_valid=False,
            schema_version=_string_or_none(raw_schema_version),
        )

    _check_unknown_extras(artifact, findings)
    _check_status_ship_semantics(artifact, findings)
    _check_walk_forward(raw, artifact, findings)
    _check_provider_capabilities(artifact, findings)
    degraded, relaxed = _check_visibility(artifact, findings)
    _check_guarded_boundary(artifact, findings)
    canonical_sha = _check_canonical_roundtrip(artifact, findings)

    semantic_valid = not any(f.severity == "ERROR" for f in findings)
    release_approved = _release_approved(artifact, semantic_valid)
    return _report(
        path=artifact_path,
        findings=findings,
        schema_valid=True,
        schema_version=artifact.schema_version,
        status=_enum_value(artifact.status),
        ship_recommendation=_enum_value(artifact.ship_recommendation),
        selected_profile=_enum_value(artifact.selected_profile),
        canonical_sha256=canonical_sha,
        degraded_capabilities=degraded,
        relaxed_safeguards=relaxed,
        guarded_boundary_crossed=artifact.guarded_boundary_crossed,
        stayed_within_guarded_boundaries=artifact.stayed_within_guarded_boundaries,
        release_approved=release_approved,
        boundary_warnings_count=len(artifact.boundary_warnings),
    )


def render_integrity_report(report: ArtifactIntegrityReport) -> str:
    """Render a compact human-readable integrity report."""

    lines = [
        "omegaprompt artifact integrity report",
        f"Artifact: {report.artifact_path}",
        "",
        "Summary:",
        f"  schema_valid: {report.schema_valid}",
        f"  semantic_valid: {report.semantic_valid}",
        f"  valid: {report.valid}",
        f"  release_approved: {report.release_approved}",
        f"  schema_version: {report.schema_version}",
        f"  status: {report.status}",
        f"  ship_recommendation: {report.ship_recommendation}",
        f"  strict_blocking_findings: {report.strict_blocking_findings}",
        "",
        "Visibility:",
        f"  degraded_capabilities: {report.counts.get('degraded_capabilities', 0)}",
        f"  relaxed_safeguards: {report.counts.get('relaxed_safeguards', 0)}",
        f"  boundary_warnings: {report.counts.get('boundary_warnings', 0)}",
    ]
    if report.canonical_sha256:
        lines.append(f"  canonical_sha256: {report.canonical_sha256}")

    lines.append("")
    if not report.findings:
        lines.append("Findings: none")
        return "\n".join(lines)

    lines.append("Findings:")
    for finding in report.findings:
        path = f" [{finding.path}]" if finding.path else ""
        lines.append(f"  - {finding.severity} {finding.id}{path}: {finding.message}")
    return "\n".join(lines)


def _check_status_ship_semantics(
    artifact: CalibrationArtifact,
    findings: list[IntegrityFinding],
) -> None:
    status = artifact.status
    ship = artifact.ship_recommendation

    if status != ArtifactStatus.OK and ship == ShipRecommendation.SHIP:
        findings.append(
            _finding(
                "NON_OK_STATUS_CANNOT_SHIP",
                "ERROR",
                "semantic",
                "A non-OK artifact cannot carry ship_recommendation='ship'.",
                details={"status": status.value, "ship_recommendation": ship.value},
            )
        )

    if status == ArtifactStatus.FAIL_KC4_GATE and ship == ShipRecommendation.SHIP:
        findings.append(
            _finding(
                "FAIL_KC4_GATE_CANNOT_SHIP",
                "ERROR",
                "semantic",
                "FAIL_KC4_GATE cannot recommend SHIP.",
                details={"status": status.value, "ship_recommendation": ship.value},
            )
        )

    if status == ArtifactStatus.OK and ship in {
        ShipRecommendation.HOLD,
        ShipRecommendation.BLOCK,
    }:
        if _has_explanation(artifact):
            findings.append(
                _finding(
                    "OK_STATUS_WITH_NON_SHIP_RECOMMENDATION",
                    "WARNING",
                    "semantic",
                    "status is OK but ship recommendation is non-ship; explanation is present and release_approved remains false.",
                    details={"ship_recommendation": ship.value},
                )
            )
        else:
            findings.append(
                _finding(
                    "OK_HOLD_BLOCK_REQUIRES_EXPLANATION",
                    "ERROR",
                    "semantic",
                    "status OK with HOLD/BLOCK must include rationale, boundary warnings, degraded capabilities, relaxed safeguards, or adaptation review classification.",
                    details={"ship_recommendation": ship.value},
                )
            )


def _check_walk_forward(
    raw: dict[str, Any],
    artifact: CalibrationArtifact,
    findings: list[IntegrityFinding],
) -> None:
    wf = artifact.walk_forward
    if wf is None:
        severity: Severity = (
            "ERROR"
            if artifact.status == ArtifactStatus.OK
            and artifact.ship_recommendation == ShipRecommendation.SHIP
            else "WARNING"
        )
        findings.append(
            _finding(
                "WALK_FORWARD_MISSING",
                severity,
                "walk_forward",
                "walk_forward is missing; holdout/KC4 integrity cannot be audited.",
                path="walk_forward",
            )
        )
        return

    raw_wf = raw.get("walk_forward")
    if not isinstance(raw_wf, dict):
        findings.append(
            _finding(
                "WALK_FORWARD_RAW_SHAPE_INVALID",
                "ERROR",
                "walk_forward",
                "walk_forward must be a JSON object before model normalization.",
                path="walk_forward",
            )
        )
        return

    required = {
        "validation_mode",
        "max_gap_threshold",
        "min_kc4_threshold",
        "shared_item_count",
        "kc4_status",
        "passed",
    }
    missing = sorted(required - set(raw_wf))
    if missing:
        findings.append(
            _finding(
                "LEGACY_WALK_FORWARD_SHAPE",
                "ERROR",
                "walk_forward",
                "walk_forward omits explanatory v2.0 fields; Pydantic defaults must not hide legacy gaps.",
                path="walk_forward",
                details={"missing": missing},
            )
        )

    if wf.kc4_status == "COMPUTED" and wf.kc4_correlation is None:
        findings.append(
            _finding(
                "KC4_COMPUTED_REQUIRES_CORRELATION",
                "ERROR",
                "walk_forward",
                "kc4_status='COMPUTED' requires non-null kc4_correlation.",
                path="walk_forward.kc4_correlation",
            )
        )

    if wf.kc4_correlation is not None and wf.min_kc4_threshold is None:
        findings.append(
            _finding(
                "KC4_THRESHOLD_MISSING_FOR_COMPUTED_VALUE",
                "ERROR",
                "walk_forward",
                "A computed kc4_correlation must record the min_kc4_threshold used by the gate.",
                path="walk_forward.min_kc4_threshold",
            )
        )

    expected_passed = _expected_walk_forward_passed(wf)
    if wf.passed != expected_passed:
        findings.append(
            _finding(
                "WALK_FORWARD_PASSED_SEMANTICS_DRIFT",
                "ERROR",
                "walk_forward",
                "walk_forward.passed does not match the recorded gap/KC4 thresholds.",
                path="walk_forward.passed",
                details={"recorded": wf.passed, "expected": expected_passed},
            )
        )

    if not wf.passed and artifact.ship_recommendation == ShipRecommendation.SHIP:
        findings.append(
            _finding(
                "FAILED_WALK_FORWARD_CANNOT_SHIP",
                "ERROR",
                "walk_forward",
                "A failed walk_forward gate cannot recommend SHIP.",
                path="ship_recommendation",
            )
        )

    if artifact.status == ArtifactStatus.FAIL_KC4_GATE and wf.passed:
        findings.append(
            _finding(
                "FAIL_KC4_STATUS_WITH_PASSED_WALK_FORWARD",
                "ERROR",
                "walk_forward",
                "status=FAIL_KC4_GATE contradicts walk_forward.passed=true.",
                path="status",
            )
        )


def _check_provider_capabilities(
    artifact: CalibrationArtifact,
    findings: list[IntegrityFinding],
) -> None:
    for role in ("target", "judge"):
        provider = getattr(artifact, f"{role}_provider")
        caps = getattr(artifact, f"{role}_capabilities")
        if provider and caps is None:
            findings.append(
                _finding(
                    f"MISSING_{role.upper()}_CAPABILITIES",
                    "ERROR",
                    "provider_capabilities",
                    f"{role}_provider is set but {role}_capabilities is null.",
                    path=f"{role}_capabilities",
                    details={f"{role}_provider": provider},
                )
            )
            continue
        if caps is None:
            findings.append(
                _finding(
                    f"{role.upper()}_CAPABILITIES_ABSENT",
                    "WARNING",
                    "provider_capabilities",
                    f"{role}_capabilities is absent; provider risk cannot be classified.",
                    path=f"{role}_capabilities",
                )
            )
            continue
        if provider and caps.provider != provider:
            findings.append(
                _finding(
                    f"{role.upper()}_CAPABILITIES_PROVIDER_MISMATCH",
                    "ERROR",
                    "provider_capabilities",
                    f"{role}_capabilities.provider does not match {role}_provider.",
                    path=f"{role}_capabilities.provider",
                    details={"provider": provider, "capabilities_provider": caps.provider},
                )
            )
        if caps.placeholder and artifact.ship_recommendation == ShipRecommendation.SHIP:
            findings.append(
                _finding(
                    f"{role.upper()}_PLACEHOLDER_CANNOT_SHIP",
                    "ERROR",
                    "provider_capabilities",
                    f"{role} provider is marked placeholder but artifact recommends ship.",
                    path=f"{role}_capabilities.placeholder",
                )
            )

    judge_caps = artifact.judge_capabilities
    if (
        artifact.selected_profile == ExecutionProfile.GUARDED
        and artifact.ship_recommendation == ShipRecommendation.SHIP
        and judge_caps is not None
        and not judge_caps.ship_grade_judge
    ):
        findings.append(
            _finding(
                "GUARDED_SHIP_REQUIRES_SHIP_GRADE_JUDGE",
                "ERROR",
                "provider_capabilities",
                "Guarded SHIP artifacts require judge_capabilities.ship_grade_judge=true.",
                path="judge_capabilities.ship_grade_judge",
            )
        )


def _check_visibility(
    artifact: CalibrationArtifact,
    findings: list[IntegrityFinding],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    degraded = [event.model_dump(mode="json") for event in artifact.degraded_capabilities]
    relaxed = [safeguard.model_dump(mode="json") for safeguard in artifact.relaxed_safeguards]
    try:
        json.dumps(degraded, sort_keys=True)
        json.dumps(relaxed, sort_keys=True)
    except (TypeError, ValueError) as exc:
        findings.append(
            _finding(
                "VISIBILITY_FIELDS_NOT_SERIALIZABLE",
                "ERROR",
                "visibility",
                f"degraded_capabilities or relaxed_safeguards are not JSON-serializable: {exc}",
            )
        )

    if degraded:
        findings.append(
            _finding(
                "DEGRADED_CAPABILITIES_VISIBLE",
                "INFO",
                "visibility",
                "degraded_capabilities are present and included in the machine-readable report.",
                path="degraded_capabilities",
                details={"count": len(degraded)},
            )
        )
    if relaxed:
        findings.append(
            _finding(
                "RELAXED_SAFEGUARDS_VISIBLE",
                "INFO",
                "visibility",
                "relaxed_safeguards are present and included in the machine-readable report.",
                path="relaxed_safeguards",
                details={"count": len(relaxed)},
            )
        )
    return degraded, relaxed


def _check_guarded_boundary(
    artifact: CalibrationArtifact,
    findings: list[IntegrityFinding],
) -> None:
    if artifact.guarded_boundary_crossed and artifact.stayed_within_guarded_boundaries:
        findings.append(
            _finding(
                "GUARDED_BOUNDARY_FLAGS_CONTRADICT",
                "ERROR",
                "guarded_boundary",
                "guarded_boundary_crossed=true contradicts stayed_within_guarded_boundaries=true.",
            )
        )

    boundary_degraded = [
        event
        for event in artifact.degraded_capabilities
        if event.affects_guarded_boundary
    ]
    if (
        artifact.selected_profile == ExecutionProfile.GUARDED
        and boundary_degraded
        and not artifact.guarded_boundary_crossed
        and artifact.stayed_within_guarded_boundaries
    ):
        findings.append(
            _finding(
                "GUARDED_DEGRADATION_HIDDEN",
                "ERROR",
                "guarded_boundary",
                "Guarded-profile capability degradation affects the guarded boundary but boundary flags remain clean.",
                path="degraded_capabilities",
                details={"count": len(boundary_degraded)},
            )
        )

    if (
        artifact.selected_profile == ExecutionProfile.GUARDED
        and boundary_degraded
        and artifact.ship_recommendation == ShipRecommendation.SHIP
    ):
        findings.append(
            _finding(
                "GUARDED_DEGRADATION_CANNOT_SHIP",
                "ERROR",
                "guarded_boundary",
                "A guarded artifact with boundary-affecting degraded capabilities cannot recommend SHIP.",
                path="ship_recommendation",
            )
        )

    if (
        artifact.selected_profile == ExecutionProfile.GUARDED
        and artifact.relaxed_safeguards
    ):
        findings.append(
            _finding(
                "RELAXED_SAFEGUARDS_IN_GUARDED_PROFILE",
                "ERROR",
                "guarded_boundary",
                "relaxed_safeguards belong to explicit boundary crossing, not a clean guarded artifact.",
                path="relaxed_safeguards",
            )
        )

    if artifact.guarded_boundary_crossed and not _has_explanation(artifact):
        findings.append(
            _finding(
                "GUARDED_BOUNDARY_CROSSING_REQUIRES_EXPLANATION",
                "ERROR",
                "guarded_boundary",
                "Guarded boundary crossing must be visible through rationale, warnings, degraded capabilities, relaxed safeguards, or adaptation review classification.",
            )
        )


def _check_unknown_extras(
    artifact: CalibrationArtifact,
    findings: list[IntegrityFinding],
) -> None:
    extra = artifact.model_extra or {}
    if extra:
        findings.append(
            _finding(
                "UNKNOWN_EXTRA_FIELDS_VISIBLE",
                "WARNING",
                "schema",
                "CalibrationArtifact accepted unknown extra fields; they are visible for review.",
                details={"fields": sorted(extra)},
            )
        )


def _check_canonical_roundtrip(
    artifact: CalibrationArtifact,
    findings: list[IntegrityFinding],
) -> str | None:
    try:
        first = canonical_artifact_json(artifact)
        second = canonical_artifact_json(CalibrationArtifact.model_validate_json(first))
    except Exception as exc:
        findings.append(
            _finding(
                "CANONICAL_JSON_ROUNDTRIP_FAILED",
                "ERROR",
                "roundtrip",
                f"Canonical JSON roundtrip failed: {exc}",
            )
        )
        return None
    if first != second:
        findings.append(
            _finding(
                "CANONICAL_JSON_ROUNDTRIP_NONDETERMINISTIC",
                "ERROR",
                "roundtrip",
                "Canonical JSON roundtrip changed bytes after re-validation.",
            )
        )
        return None
    return hashlib.sha256(first.encode("utf-8")).hexdigest()


def _expected_walk_forward_passed(wf: Any) -> bool:
    gap_ok = wf.generalization_gap <= wf.max_gap_threshold
    if wf.validation_mode == "paired" and wf.kc4_correlation is None:
        kc4_ok = False
    elif wf.kc4_correlation is None or wf.min_kc4_threshold is None:
        kc4_ok = True
    else:
        kc4_ok = wf.kc4_correlation >= wf.min_kc4_threshold
    return bool(gap_ok and kc4_ok)


def _has_explanation(artifact: CalibrationArtifact) -> bool:
    if artifact.rationale.strip():
        return True
    if artifact.boundary_warnings or artifact.degraded_capabilities or artifact.relaxed_safeguards:
        return True
    summary = artifact.adaptation_summary or {}
    return bool(
        summary.get("manual_review_required")
        or summary.get("manual_review_reasons")
        or summary.get("advisory_not_applied")
    )


def _release_approved(artifact: CalibrationArtifact, semantic_valid: bool) -> bool:
    return bool(
        semantic_valid
        and artifact.status == ArtifactStatus.OK
        and artifact.ship_recommendation == ShipRecommendation.SHIP
        and artifact.stayed_within_guarded_boundaries
        and not artifact.guarded_boundary_crossed
        and not artifact.relaxed_safeguards
    )


def canonical_artifact_json(artifact: CalibrationArtifact) -> str:
    """Return deterministic normalized JSON for a validated artifact."""

    return json.dumps(
        artifact.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def normalized_artifact_hash(artifact: CalibrationArtifact) -> str:
    """Return the SHA-256 hash of ``canonical_artifact_json(artifact)``."""

    return hashlib.sha256(canonical_artifact_json(artifact).encode("utf-8")).hexdigest()


def _report(
    *,
    path: Path,
    findings: list[IntegrityFinding],
    schema_valid: bool,
    schema_version: str | None = None,
    status: str | None = None,
    ship_recommendation: str | None = None,
    selected_profile: str | None = None,
    canonical_sha256: str | None = None,
    degraded_capabilities: list[dict[str, Any]] | None = None,
    relaxed_safeguards: list[dict[str, Any]] | None = None,
    guarded_boundary_crossed: bool | None = None,
    stayed_within_guarded_boundaries: bool | None = None,
    release_approved: bool = False,
    boundary_warnings_count: int = 0,
) -> ArtifactIntegrityReport:
    effective_schema_valid = schema_valid and not any(
        f.severity == "ERROR" and f.category == "schema" for f in findings
    )
    semantic_valid = effective_schema_valid and not any(
        f.severity == "ERROR" for f in findings
    )
    degraded = degraded_capabilities or []
    relaxed = relaxed_safeguards or []
    counts = {
        "findings": len(findings),
        "errors": sum(1 for finding in findings if finding.severity == "ERROR"),
        "warnings": sum(1 for finding in findings if finding.severity == "WARNING"),
        "infos": sum(1 for finding in findings if finding.severity == "INFO"),
        "degraded_capabilities": len(degraded),
        "relaxed_safeguards": len(relaxed),
        "boundary_warnings": boundary_warnings_count,
    }
    return ArtifactIntegrityReport(
        artifact_path=str(path),
        schema_valid=effective_schema_valid,
        semantic_valid=semantic_valid,
        valid=effective_schema_valid and semantic_valid,
        release_approved=release_approved and effective_schema_valid and semantic_valid,
        schema_version=schema_version,
        status=status,
        ship_recommendation=ship_recommendation,
        selected_profile=selected_profile,
        canonical_sha256=canonical_sha256,
        degraded_capabilities=degraded,
        relaxed_safeguards=relaxed,
        guarded_boundary_crossed=guarded_boundary_crossed,
        stayed_within_guarded_boundaries=stayed_within_guarded_boundaries,
        counts=counts,
        findings=findings,
    )


def _finding(
    finding_id: str,
    severity: Severity,
    category: str,
    message: str,
    *,
    path: str | None = None,
    details: dict[str, Any] | None = None,
) -> IntegrityFinding:
    return IntegrityFinding(
        id=finding_id,
        severity=severity,
        category=category,
        message=message,
        path=path,
        details=details or {},
    )


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
