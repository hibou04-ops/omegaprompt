"""Eval and artifact result schemas.

These are the write side of the calibration pipeline. Every file downstream
of a calibration run reads a ``CalibrationArtifact`` (or a fragment), so
this contract has to stay stable across minor versions within v1.x.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from omegaprompt.domain.judge import JudgeResult


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


class EvalResult(BaseModel):
    """Aggregate result for one evaluation pass (one param set, whole dataset).

    This is what a ``CalibrableTarget.evaluate`` returns. The omega-lock
    search layer consumes ``fitness``; everything else is for reporting.
    """

    model_config = ConfigDict(extra="allow")

    params: dict
    item_results: list[EvalItemResult]
    fitness: float
    hard_gate_pass_rate: float = 0.0
    usage_summary: dict[str, int] = Field(default_factory=dict)


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

    schema_version: str = "1.0"
    method: str = Field(..., description="Search method used (p1 / grid / tpe / ...).")
    unlock_k: int = Field(..., ge=0)
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

    target_provider: str | None = None
    target_model: str | None = None
    judge_provider: str | None = None
    judge_model: str | None = None

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
