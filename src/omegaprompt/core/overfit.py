"""Overfit-metric extraction — the "is my prompt overfit?" number.

The data that answers "did my calibrated prompt generalize, or did I just
overfit the training slice?" already lives inside
:class:`~omegaprompt.domain.result.WalkForwardResult`:

* ``kc4_correlation`` — the train<->holdout *transfer* correlation
  (per-item Pearson r between the train and test slices). High = the
  candidate's per-item behaviour transfers; low/negative = it overfit.
* ``generalization_gap`` — the normalized train-vs-test fitness gap.
  Small = stable; large = the candidate looks good on train but
  degrades on the holdout.

This module surfaces those two numbers as one prominent, machine-readable
summary so CI / agents do not have to dig through the nested walk-forward
block. It is intentionally a *pure read* over an existing artifact: it
adds **no** field to ``CalibrationArtifact``, so the artifact schema stays
at ``"2.0"`` and every byte-stable golden hash is preserved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from omegaprompt.domain.result import CalibrationArtifact


# Schema version for the standalone overfit summary payload. Independent of
# the artifact schema ("2.0") so this surface can evolve without touching
# the frozen artifact contract or its golden hashes.
OVERFIT_SUMMARY_SCHEMA_VERSION = "1.0"


class OverfitMetrics(BaseModel):
    """Machine-readable train<->holdout transfer summary.

    A standalone payload (not an artifact field) so it can be embedded in
    ``report --format json`` and the ``gate`` command output without
    mutating the artifact schema.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = OVERFIT_SUMMARY_SCHEMA_VERSION

    available: bool = Field(
        default=False,
        description="True when a walk_forward block was present to read.",
    )
    transfer_correlation: float | None = Field(
        default=None,
        description=(
            "Train<->holdout per-item Pearson r (KC-4). None when KC-4 "
            "was not computed; ``transfer_correlation_status`` says why."
        ),
    )
    transfer_correlation_status: str | None = Field(
        default=None,
        description="kc4_status from the walk-forward block (e.g. COMPUTED).",
    )
    min_transfer_correlation_threshold: float | None = Field(
        default=None,
        description="The pre-declared min_kc4 the gate checked against.",
    )
    generalization_gap: float | None = Field(
        default=None,
        description="Normalized |train - test| / |train| fitness gap.",
    )
    generalization_gap_status: str | None = Field(
        default=None,
        description="gap_status from the walk-forward block (e.g. OK).",
    )
    max_generalization_gap_threshold: float | None = Field(
        default=None,
        description="The pre-declared max_gap the gate checked against.",
    )
    validation_mode: str | None = Field(
        default=None,
        description="Which KC-4 contract was declared (auto/paired/disjoint).",
    )
    shared_item_count: int | None = Field(
        default=None,
        description="How many item ids the train and test slices shared.",
    )
    train_fitness: float | None = None
    test_fitness: float | None = None
    walk_forward_passed: bool | None = Field(
        default=None,
        description="Did the candidate clear the pre-declared gap + KC-4 gate?",
    )
    overfit_verdict: str = Field(
        default="UNKNOWN",
        description=(
            "Coarse, deterministic read for dashboards: GENERALIZES (gate "
            "passed), OVERFIT (gate failed on gap/correlation), "
            "UNVERIFIABLE (no walk-forward / KC-4 not computed), or "
            "UNKNOWN (no data)."
        ),
    )


def extract_overfit_metrics(artifact: "CalibrationArtifact") -> OverfitMetrics:
    """Extract the train<->holdout transfer numbers from an artifact.

    Pure read. Returns ``available=False`` (verdict ``UNKNOWN``) when the
    artifact carries no walk-forward block.
    """

    wf = getattr(artifact, "walk_forward", None)
    if wf is None:
        return OverfitMetrics(available=False, overfit_verdict="UNKNOWN")

    corr = wf.kc4_correlation
    corr_status = wf.kc4_status
    gap = wf.generalization_gap
    passed = bool(wf.passed)

    verdict = _verdict(
        passed=passed,
        corr_status=corr_status,
        gap=gap,
        max_gap=wf.max_gap_threshold,
    )

    return OverfitMetrics(
        available=True,
        transfer_correlation=corr,
        transfer_correlation_status=corr_status,
        min_transfer_correlation_threshold=wf.min_kc4_threshold,
        generalization_gap=gap,
        generalization_gap_status=wf.gap_status,
        max_generalization_gap_threshold=wf.max_gap_threshold,
        validation_mode=wf.validation_mode,
        shared_item_count=wf.shared_item_count,
        train_fitness=wf.train_best_fitness,
        test_fitness=wf.test_fitness,
        walk_forward_passed=passed,
        overfit_verdict=verdict,
    )


def _verdict(*, passed: bool, corr_status: str, gap: float, max_gap: float) -> str:
    """Coarse, deterministic overfit read.

    * ``GENERALIZES`` — the pre-declared walk-forward gate passed.
    * ``OVERFIT`` — the gate failed (gap exceeded threshold and/or the
      transfer correlation fell below its threshold). Either arm is an
      overfit signal.
    * ``UNVERIFIABLE`` — the gate did not pass but neither failure arm
      can be substantiated: the correlation was not computed AND the gap
      is within its threshold. The most defensible read is "we could not
      verify generalization", not a confident OVERFIT claim.
    """

    if passed:
        return "GENERALIZES"
    gap_failed = gap > max_gap
    corr_blind = corr_status != "COMPUTED"
    if gap_failed or corr_status == "COMPUTED":
        return "OVERFIT"
    if corr_blind:
        return "UNVERIFIABLE"
    return "OVERFIT"


def overfit_metrics_dict(artifact: "CalibrationArtifact") -> dict[str, Any]:
    """Convenience: ``extract_overfit_metrics`` as a JSON-ready dict."""

    return extract_overfit_metrics(artifact).model_dump(mode="json")
