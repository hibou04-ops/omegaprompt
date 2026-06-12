"""Ship-gate logic — the dedicated CI "can I ship this?" verdict.

``omegaprompt gate`` is a first-class CI gate. It fuses two existing
checks into one pass/fail decision so shipping no longer has to be
inferred from ``diff`` / ``report``:

1. **Artifact integrity** — the same zero-network audit
   (:func:`omegaprompt.core.artifact_integrity.check_artifact_integrity`)
   that ``check-artifact`` runs: schema validity, status/ship coherence,
   walk-forward shape, provider-capability coherence, canonical roundtrip.

2. **Holdout transfer / gap verdict** — the train<->holdout
   generalization read (:func:`omegaprompt.core.overfit.extract_overfit_metrics`):
   did the candidate clear its pre-declared walk-forward gate, or did it
   overfit the training slice?

The gate is deterministic and offline: no provider calls, no network. The
JSON summary is schema-versioned so CI can consume it stably.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from omegaprompt.core.artifact_integrity import check_artifact_integrity
from omegaprompt.core.overfit import OverfitMetrics, extract_overfit_metrics

# Versioned independently of the artifact schema so the gate summary can
# evolve without touching the frozen artifact contract.
GATE_SCHEMA_VERSION = "1.0"


class GateResult(BaseModel):
    """Machine-readable ship-gate verdict."""

    model_config = ConfigDict(extra="forbid")

    gate_schema_version: str = GATE_SCHEMA_VERSION
    artifact_path: str

    passed: bool = Field(
        ...,
        description="True only when every required gate condition holds.",
    )
    exit_code: int = Field(
        ...,
        description="0 when passed; 1 on a ship-blocking condition; 2 on environment/load failure.",
    )

    # Constituent verdicts.
    integrity_valid: bool = Field(
        ...,
        description="Artifact passed schema + semantic integrity (no ERROR findings).",
    )
    release_approved: bool = Field(
        ...,
        description="Integrity checker's release_approved (OK status + ship + clean boundaries).",
    )
    status: str | None = None
    ship_recommendation: str | None = None
    overfit_verdict: str = Field(
        default="UNKNOWN",
        description="GENERALIZES / OVERFIT / UNVERIFIABLE / UNKNOWN.",
    )
    walk_forward_passed: bool | None = None

    blocking_reasons: list[str] = Field(default_factory=list)
    integrity_error_count: int = 0
    overfit: OverfitMetrics = Field(default_factory=OverfitMetrics)

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def run_gate(
    artifact_path: str | Path,
    *,
    require_generalization: bool = True,
) -> GateResult:
    """Evaluate the ship gate for an artifact at ``artifact_path``.

    Args:
        artifact_path: Path to a CalibrationArtifact JSON.
        require_generalization: When True (default), an absent or
            unverifiable walk-forward verdict blocks the gate — a ship
            decision should not be made blind to holdout transfer. Set
            False to gate on integrity + release-approval only (the
            walk-forward read is still reported, just not blocking).

    Returns:
        A :class:`GateResult`. ``exit_code`` is 0/1/2 for
        pass / ship-blocked / environment-or-load-failure.
    """

    artifact_path = Path(artifact_path)
    report = check_artifact_integrity(artifact_path)

    blocking: list[str] = []

    # Environment / load failure is exit 2, distinct from a clean
    # ship-blocking verdict (exit 1).
    if report.environment_blocked:
        return GateResult(
            artifact_path=str(artifact_path),
            passed=False,
            exit_code=2,
            integrity_valid=False,
            release_approved=False,
            status=report.status,
            ship_recommendation=report.ship_recommendation,
            overfit_verdict="UNKNOWN",
            walk_forward_passed=None,
            blocking_reasons=["artifact could not be loaded (environment blocked)"],
            integrity_error_count=report.strict_blocking_findings,
            overfit=OverfitMetrics(available=False, overfit_verdict="UNKNOWN"),
        )

    integrity_valid = bool(report.valid)
    if not integrity_valid:
        blocking.append(
            f"artifact integrity failed ({report.strict_blocking_findings} ERROR finding(s))"
        )

    if not report.release_approved:
        blocking.append(
            "artifact is not release-approved "
            f"(status={report.status}, ship={report.ship_recommendation})"
        )

    # Overfit / holdout transfer read. We need a loadable artifact to read
    # walk-forward; only reachable here because integrity loaded the model.
    overfit = _extract_overfit_safe(artifact_path)

    if require_generalization:
        if not overfit.available:
            blocking.append("no walk-forward block — generalization is unmeasured")
        elif overfit.overfit_verdict == "OVERFIT":
            blocking.append("walk-forward gate failed — candidate overfit the training slice")
        elif overfit.overfit_verdict == "UNVERIFIABLE":
            blocking.append("walk-forward gate did not pass and could not verify transfer")

    passed = not blocking
    exit_code = 0 if passed else 1

    return GateResult(
        artifact_path=str(artifact_path),
        passed=passed,
        exit_code=exit_code,
        integrity_valid=integrity_valid,
        release_approved=bool(report.release_approved),
        status=report.status,
        ship_recommendation=report.ship_recommendation,
        overfit_verdict=overfit.overfit_verdict,
        walk_forward_passed=overfit.walk_forward_passed,
        blocking_reasons=blocking,
        integrity_error_count=report.strict_blocking_findings,
        overfit=overfit,
    )


def _extract_overfit_safe(artifact_path: Path) -> OverfitMetrics:
    """Load the artifact and read overfit metrics; fail soft to UNKNOWN."""

    from omegaprompt.core.artifact import load_artifact

    try:
        artifact = load_artifact(artifact_path)
    except Exception:
        return OverfitMetrics(available=False, overfit_verdict="UNKNOWN")
    return extract_overfit_metrics(artifact)


def render_gate_report(result: GateResult) -> str:
    """Compact human-readable gate report."""

    lines = [
        "omegaprompt ship gate",
        f"Artifact: {result.artifact_path}",
        "",
        f"  PASSED: {result.passed}",
        f"  exit_code: {result.exit_code}",
        f"  integrity_valid: {result.integrity_valid}",
        f"  release_approved: {result.release_approved}",
        f"  status: {result.status}",
        f"  ship_recommendation: {result.ship_recommendation}",
        f"  overfit_verdict: {result.overfit_verdict}",
        f"  walk_forward_passed: {result.walk_forward_passed}",
    ]
    if result.overfit.available:
        lines.append(
            "  transfer_correlation: "
            + (
                f"{result.overfit.transfer_correlation}"
                if result.overfit.transfer_correlation is not None
                else f"not computed ({result.overfit.transfer_correlation_status})"
            )
        )
        lines.append(f"  generalization_gap: {result.overfit.generalization_gap}")
    lines.append("")
    if result.blocking_reasons:
        lines.append("Blocking reasons:")
        for reason in result.blocking_reasons:
            lines.append(f"  - {reason}")
    else:
        lines.append("No blocking reasons — clear to ship.")
    return "\n".join(lines)
