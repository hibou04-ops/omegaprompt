"""Provider-neutral prompt parameter space and variants.

``PromptVariants`` holds the concrete pools the searcher samples indices
into (system prompts, few-shot examples).

``MetaAxisSpace`` declares the bounds for each meta-axis. The calibration
searcher emits a parameter dict whose keys are axis names and values are
ordinals or enum members; ``ResolvedPromptParams`` is the structured form
after adapter translation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omegaprompt.domain.enums import (
    OutputBudgetBucket,
    ReasoningProfile,
    ResponseSchemaMode,
    ToolPolicyVariant,
)


class PromptVariants(BaseModel):
    """User-supplied discrete pools of prompt components.

    The searcher picks an index into ``system_prompts`` and a count into
    ``few_shot_examples``. Both axes are first-class in every calibration
    run and cannot be fully disabled (set ``len(system_prompts) == 1`` to
    effectively freeze the system prompt).
    """

    model_config = ConfigDict(extra="forbid")

    system_prompts: list[str] = Field(
        default_factory=list,
        description="Ordered list of candidate system prompt strings.",
    )
    few_shot_examples: list[dict[str, str]] = Field(
        default_factory=list,
        description="Pool of {input, output} dicts used as few-shot demos.",
    )

    @field_validator("system_prompts")
    @classmethod
    def _non_empty_variants(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("PromptVariants.system_prompts must contain at least one prompt.")
        if any(not isinstance(s, str) or not s.strip() for s in v):
            raise ValueError("Every system prompt must be a non-empty string.")
        return v


class MetaAxisSpace(BaseModel):
    """Declarative bounds for every meta-axis.

    The calibration searcher reads this to decide which values to probe.
    Every axis supports a list of allowed values; an empty list means the
    axis is disabled (only the neutral value is used).
    """

    model_config = ConfigDict(extra="forbid")

    system_prompt_idx_max: int = Field(
        ...,
        ge=0,
        description="Upper bound (inclusive) for the system_prompt index. "
        "Typically len(variants.system_prompts) - 1.",
    )
    few_shot_min: int = Field(default=0, ge=0)
    few_shot_max: int = Field(default=3, ge=0)

    reasoning_profiles: list[ReasoningProfile] = Field(
        default_factory=lambda: [
            ReasoningProfile.OFF,
            ReasoningProfile.STANDARD,
            ReasoningProfile.DEEP,
        ],
        description="Which reasoning profiles the searcher may pick from.",
    )
    output_budgets: list[OutputBudgetBucket] = Field(
        default_factory=lambda: [
            OutputBudgetBucket.SMALL,
            OutputBudgetBucket.MEDIUM,
            OutputBudgetBucket.LARGE,
        ],
    )
    response_schema_modes: list[ResponseSchemaMode] = Field(
        default_factory=lambda: [ResponseSchemaMode.FREEFORM],
        description="Target response shape. Only unlock >1 value when the "
        "target semantics actually vary with schema mode.",
    )
    tool_policy_variants: list[ToolPolicyVariant] = Field(
        default_factory=lambda: [ToolPolicyVariant.NO_TOOLS],
        description="Tool-use policy. Leave at NO_TOOLS for plain chat targets.",
    )

    @field_validator("few_shot_max")
    @classmethod
    def _few_shot_range(cls, v: int, info: Any) -> int:
        few_shot_min = info.data.get("few_shot_min", 0)
        if v < few_shot_min:
            raise ValueError(f"few_shot_max ({v}) must be >= few_shot_min ({few_shot_min}).")
        return v

    @field_validator(
        "reasoning_profiles",
        "output_budgets",
        "response_schema_modes",
        "tool_policy_variants",
    )
    @classmethod
    def _non_empty(cls, v: list, info: Any) -> list:
        if not v:
            raise ValueError(f"{info.field_name} must have at least one member.")
        return v

    def axis_names(self) -> list[str]:
        """Names of the meta-axes, in a stable order."""
        return [
            "system_prompt_idx",
            "few_shot_count",
            "reasoning_profile",
            "output_budget",
            "response_schema_mode",
            "tool_policy",
        ]


class ResolvedPromptParams(BaseModel):
    """The concrete parameter set after the searcher's choices are resolved.

    This is what a target adapter consumes. It is semi-typed: enums are
    preserved, not flattened to strings, so the adapter can dispatch
    cleanly.
    """

    model_config = ConfigDict(extra="forbid")

    system_prompt_idx: int
    few_shot_count: int
    reasoning_profile: ReasoningProfile = ReasoningProfile.STANDARD
    output_budget: OutputBudgetBucket = OutputBudgetBucket.MEDIUM
    response_schema_mode: ResponseSchemaMode = ResponseSchemaMode.FREEFORM
    tool_policy: ToolPolicyVariant = ToolPolicyVariant.NO_TOOLS
