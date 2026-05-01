"""Eval and artifact result schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from omegaprompt.domain.judge import JudgeResult
from omegaprompt.domain.profiles import (
    BoundaryWarning,
    ExecutionProfile,
    RelaxedSafeguard,
    ShipRecommendation,
)
from omegaprompt.providers.base import CapabilityEvent, ProviderCapabilities


class PerItemScore(BaseModel):
    """Fitness breakdown for a single dataset item after the judge runs."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    soft_score: float
    gates_passed: bool
    final_score: float
    notes: str = ""


class EvalItemResult(BaseModel):
    """A single target+judge roundtrip result for one dataset item."""

    model_config = ConfigDict(extra="allow")

    item_id: str
    params: dict
    raw_output: str
    judge: JudgeResult
    token_usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: float = 0.0
    degraded_capabilities: list[CapabilityEvent] = Field(default_factory=list)
    boundary_warnings: list[BoundaryWarning] = Field(default_factory=list)


class EvalResult(BaseModel):
    """Aggregate result for one evaluation pass (one param set, whole dataset).

    This is what a ``CalibrableTarget.evaluate`` returns. The omega-lock
    search layer consumes ``fitness``; everything else is for reporting.
    """

    model_config = ConfigDict(extra="allow")

    params: dict
    resolved_params: dict[str, Any] = Field(default_factory=dict)
    item_results: list[EvalItemResult]
    fitness: float
    n_trials: int
    hard_gate_pass_rate: float = 0.0
    usage_summary: dict[str, int] = Field(default_factory=dict)
    latency_ms: float = 0.0
    estimated_cost_units: float = 0.0
    degraded_capabilities: list[CapabilityEvent] = Field(default_factory=list)
    boundary_warnings: list[BoundaryWarning] = Field(default_factory=list)
    within_guarded_boundaries: bool = True
    ship_recommendation: ShipRecommendation = ShipRecommendation.HOLD
    metadata: dict[str, Any] = Field(default_factory=dict)


class WalkForwardResult(BaseModel):
    """Train/test generalization assessment for one candidate.

    The result preserves both the numbers (fitness, gap, KC-4) and the
    *conditions* under which they were computed (validation_mode, shared
    item count, status enums, declared thresholds). A reviewer reading
    this artifact can tell whether KC-4 was None because the split was
    structurally disjoint, the per-item scores had zero variance, or
    the slices shared too few ids — three cases the old shape collapsed
    into a single ``None`` (Reviewer P1 #6).
    """

    model_config = ConfigDict(extra="forbid")

    train_best_fitness: float
    test_fitness: float
    generalization_gap: float = Field(
        ...,
        description=(
            "|train - test| / |train|. Smaller is better; 0 = no drift. "
            "When train_best_fitness is 0 the gap is reported as 1.0 to "
            "stay JSON-friendly; the structural meaning is preserved on "
            "``gap_status``."
        ),
    )
    gap_status: Literal[
        "OK",
        "TRAIN_ZERO_BOTH_ZERO",
        "TRAIN_ZERO_TEST_NONZERO",
    ] = Field(
        default="OK",
        description=(
            "Why the gap reads what it does. ``OK`` for the normal case. "
            "``TRAIN_ZERO_BOTH_ZERO`` when both slices scored zero (gap "
            "is structurally undefined). ``TRAIN_ZERO_TEST_NONZERO`` "
            "when train was zero but test was not (denominator is zero, "
            "uplift is technically infinite). Reviewer P1 #8: the old "
            "shape squashed both cases into 1.0."
        ),
    )

    validation_mode: Literal["auto", "paired", "disjoint"] = Field(
        default="auto",
        description=(
            "Which KC-4 contract the caller declared. ``auto`` preserves "
            "historical behaviour (compute when slices share >=3 ids, "
            "otherwise skip). ``paired`` is strict: zero-variance / "
            "fewer than 3 shared ids fail closed. ``disjoint`` skips "
            "KC-4 by design — the gate is gap-only."
        ),
    )
    shared_item_count: int = Field(
        default=0,
        ge=0,
        description="How many item ids the train and test slices share.",
    )

    kc4_correlation: float | None = Field(
        default=None,
        description=(
            "Pearson correlation between per-item scores on train and "
            "test, or None when the correlation could not be computed. "
            "``kc4_status`` records *why* it is None."
        ),
    )
    kc4_status: Literal[
        "COMPUTED",
        "NOT_APPLICABLE_DISJOINT",
        "INSUFFICIENT_SHARED_ITEMS",
        "MISSING_PER_ITEM_SCORES",
        "ZERO_VARIANCE_TRAIN",
        "ZERO_VARIANCE_TEST",
        "ZERO_VARIANCE_BOTH",
        "PEARSON_NAN",
    ] = Field(
        default="INSUFFICIENT_SHARED_ITEMS",
        description=(
            "Outcome of the KC-4 computation. ``COMPUTED`` is the only "
            "value where ``kc4_correlation`` is non-None and the gate "
            "actually checked. The others tell a reviewer whether KC-4 "
            "was unmeasurable for structural reasons (disjoint split, "
            "missing scores) or because the data degenerated (zero "
            "variance, NaN). Reviewer P1 #6."
        ),
    )

    max_gap_threshold: float = Field(
        default=0.25,
        description=(
            "The pre-declared max_gap that produced the pass/fail "
            "verdict. Recorded so a future reader can tell which "
            "threshold was active without rerunning."
        ),
    )
    min_kc4_threshold: float | None = Field(
        default=0.5,
        description=(
            "The pre-declared min_kc4. None when KC-4 was not "
            "applicable (disjoint split or auto with no overlap)."
        ),
    )

    passed: bool = Field(
        ...,
        description="Did the candidate clear the pre-declared gap + KC-4 thresholds?",
    )


class CalibrationArtifact(BaseModel):
    """The machine-readable output of a full calibration run.

    v1.0 supersedes v0.2's ``CalibrationOutcome`` with a richer schema that
    carries the full per-axis sensitivity ranking, the resolved meta-axis
    ``best_params`` (enum values, not integer indices), and an explicit
    walk-forward result block.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: str = "2.0"
    engine_name: str = "omegaprompt"
    method: str = Field(..., description="Search method used (p1 / grid / tpe / ...).")
    unlock_k: int = Field(..., ge=0)
    selected_profile: ExecutionProfile = ExecutionProfile.GUARDED
    neutral_baseline_params: dict[str, Any] = Field(default_factory=dict)
    calibrated_params: dict[str, Any] = Field(default_factory=dict)
    neutral_fitness: float = 0.0
    calibrated_fitness: float = 0.0
    uplift_absolute: float = 0.0
    uplift_percent: float = 0.0
    quality_per_cost_neutral: float = 0.0
    quality_per_cost_best: float = 0.0
    quality_per_latency_neutral: float = 0.0
    quality_per_latency_best: float = 0.0
    boundary_warnings: list[BoundaryWarning] = Field(default_factory=list)
    degraded_capabilities: list[CapabilityEvent] = Field(default_factory=list)
    ship_recommendation: ShipRecommendation = ShipRecommendation.HOLD
    stayed_within_guarded_boundaries: bool = True
    additional_uplift_from_boundary_crossing: float = 0.0
    relaxed_safeguards: list[RelaxedSafeguard] = Field(default_factory=list)
    guarded_boundary_crossed: bool = False
    cost_basis: str = Field(
        default="normalized_token_units",
        description="Uses adapter pricing when available; otherwise token-normalized units.",
    )
    best_params: dict[str, Any] = Field(
        ...,
        description="Best meta-axis parameters. Enum values are serialised as their .value.",
    )
    best_fitness: float
    walk_forward: WalkForwardResult | None = Field(default=None)
    hard_gate_pass_rate: float = Field(..., ge=0.0, le=1.0)

    sensitivity_ranking: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered list of {axis, gini_delta, rank} from highest to lowest stress.",
    )

    n_candidates_evaluated: int = Field(..., ge=0)
    total_api_calls: int = Field(..., ge=0)
    usage_summary: dict[str, int] = Field(default_factory=dict)
    latency_summary_ms: dict[str, float] = Field(default_factory=dict)

    target_provider: str | None = None
    target_model: str | None = None
    judge_provider: str | None = None
    judge_model: str | None = None
    target_capabilities: ProviderCapabilities | None = None
    judge_capabilities: ProviderCapabilities | None = None

    status: str = Field(
        default="OK",
        description=(
            "One of: OK, FAIL_KC4_GATE, FAIL_HARD_GATES, FAIL_NO_CANDIDATES, "
            "REQUIRES_MANUAL_REVIEW. Downstream CI checks status first."
        ),
    )
    rationale: str = Field(
        default="",
        description="One-sentence human-readable summary of why status is what it is.",
    )
    adaptation_summary: dict[str, Any] | None = Field(
        default=None,
        description=(
            "When an adaptation_plan was supplied to calibrate(), summary of "
            "which overrides actually flowed through the pipeline vs which "
            "were recorded for reviewer awareness only. Keys: ``applied`` "
            "(list of parameter names that reached the search loop or the "
            "ship gate), ``advisory_not_applied`` (recorded but no consumer), "
            "``manual_review_required`` (bool), ``manual_review_reasons`` "
            "(list[str]). Reviewer P0: makes the artifact honest about "
            "which adaptation_plan claims actually changed pipeline behaviour."
        ),
    )

    def model_post_init(self, __context: Any) -> None:
        if not self.calibrated_params:
            self.calibrated_params = dict(self.best_params)
        if not self.best_params:
            self.best_params = dict(self.calibrated_params)
        if self.calibrated_fitness == 0.0 and self.best_fitness:
            self.calibrated_fitness = self.best_fitness
        if self.best_fitness == 0.0 and self.calibrated_fitness:
            self.best_fitness = self.calibrated_fitness
