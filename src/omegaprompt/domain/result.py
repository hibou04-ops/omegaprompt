"""Eval and artifact result schemas."""

from __future__ import annotations

from typing import Any

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
    """Train/test generalization assessment for one candidate."""

    model_config = ConfigDict(extra="forbid")

    train_best_fitness: float
    test_fitness: float
    generalization_gap: float = Field(
        ...,
        description="|train - test| / |train|. Smaller is better; 0 = no drift.",
    )
    kc4_correlation: float | None = Field(
        default=None,
        description="Pearson correlation between per-item scores on train and test."
        " None when the slices do not share item ids.",
    )
    passed: bool = Field(
        ...,
        description="Did the candidate clear the pre-declared KC-4 threshold?",
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
            "One of: OK, FAIL_KC4_GATE, FAIL_HARD_GATES, FAIL_NO_CANDIDATES. "
            "Downstream CI checks status first."
        ),
    )
    rationale: str = Field(
        default="",
        description="One-sentence human-readable summary of why status is what it is.",
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
