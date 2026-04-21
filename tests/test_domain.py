"""v1.0 domain-layer coverage: enums, MetaAxisSpace, ResolvedPromptParams."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omegaprompt.domain.enums import (
    OUTPUT_BUDGET_ORDINALS,
    OutputBudgetBucket,
    REASONING_ORDINALS,
    ReasoningProfile,
    ResponseSchemaMode,
    ToolPolicyVariant,
    output_budget_from_ordinal,
    reasoning_from_ordinal,
)
from omegaprompt.domain.params import MetaAxisSpace, ResolvedPromptParams


def test_reasoning_ordinals_ascending():
    ordered = sorted(REASONING_ORDINALS.items(), key=lambda kv: kv[1])
    assert [p for p, _ in ordered] == [
        ReasoningProfile.OFF,
        ReasoningProfile.LIGHT,
        ReasoningProfile.STANDARD,
        ReasoningProfile.DEEP,
    ]


def test_output_budget_ordinals_ascending():
    ordered = sorted(OUTPUT_BUDGET_ORDINALS.items(), key=lambda kv: kv[1])
    assert [b for b, _ in ordered] == [
        OutputBudgetBucket.SMALL,
        OutputBudgetBucket.MEDIUM,
        OutputBudgetBucket.LARGE,
    ]


def test_reasoning_from_ordinal_clamps():
    assert reasoning_from_ordinal(-99) == ReasoningProfile.OFF
    assert reasoning_from_ordinal(99) == ReasoningProfile.DEEP


def test_output_budget_from_ordinal_clamps():
    assert output_budget_from_ordinal(-1) == OutputBudgetBucket.SMALL
    assert output_budget_from_ordinal(5) == OutputBudgetBucket.LARGE


def test_meta_axis_space_axis_names_stable():
    s = MetaAxisSpace(system_prompt_idx_max=2)
    assert s.axis_names() == [
        "system_prompt_idx",
        "few_shot_count",
        "reasoning_profile",
        "output_budget",
        "response_schema_mode",
        "tool_policy",
    ]


def test_resolved_prompt_params_defaults():
    r = ResolvedPromptParams(system_prompt_idx=0, few_shot_count=0)
    assert r.reasoning_profile == ReasoningProfile.STANDARD
    assert r.output_budget == OutputBudgetBucket.MEDIUM
    assert r.response_schema_mode == ResponseSchemaMode.FREEFORM
    assert r.tool_policy == ToolPolicyVariant.NO_TOOLS


def test_resolved_prompt_params_forbids_extra():
    with pytest.raises(ValidationError):
        ResolvedPromptParams(
            system_prompt_idx=0,
            few_shot_count=0,
            unknown_axis="bad",
        )
