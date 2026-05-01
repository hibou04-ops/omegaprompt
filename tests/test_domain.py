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
from omegaprompt.domain.params import (
    MetaAxisSpace,
    PromptVariants,
    ResolvedPromptParams,
    validate_space_against_variants,
)


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
        "system_prompt_variant",
        "few_shot_count",
        "reasoning_profile",
        "output_budget_bucket",
        "response_schema_mode",
        "tool_policy_variant",
    ]


def test_resolved_prompt_params_defaults():
    r = ResolvedPromptParams(system_prompt_variant=0, few_shot_count=0)
    assert r.reasoning_profile == ReasoningProfile.STANDARD
    assert r.output_budget_bucket == OutputBudgetBucket.MEDIUM
    assert r.response_schema_mode == ResponseSchemaMode.FREEFORM
    assert r.tool_policy_variant == ToolPolicyVariant.NO_TOOLS
    # Compatibility aliases remain readable during migration.
    assert r.system_prompt_idx == 0
    assert r.output_budget == OutputBudgetBucket.MEDIUM
    assert r.tool_policy == ToolPolicyVariant.NO_TOOLS


def test_resolved_prompt_params_forbids_extra():
    with pytest.raises(ValidationError):
        ResolvedPromptParams(
            system_prompt_variant=0,
            few_shot_count=0,
            unknown_axis="bad",
        )


# ---------------------------------------------------------------------------
# Reviewer P1 #10: PromptVariants.few_shot_examples must validate the
# {input, output} shape. Provider _messages() helpers index ``shot["input"]``
# and ``shot["output"]`` directly; a malformed shot would either KeyError
# mid-eval or silently send an empty turn that confuses the target.
# ---------------------------------------------------------------------------


def test_prompt_variants_few_shot_requires_input_and_output_keys():
    with pytest.raises(ValidationError, match="few_shot_examples"):
        PromptVariants(
            system_prompts=["sp"],
            few_shot_examples=[{"input": "q1"}],  # missing output
        )
    with pytest.raises(ValidationError, match="few_shot_examples"):
        PromptVariants(
            system_prompts=["sp"],
            few_shot_examples=[{"output": "a1"}],  # missing input
        )


def test_prompt_variants_few_shot_rejects_empty_string_values():
    with pytest.raises(ValidationError, match="non-empty"):
        PromptVariants(
            system_prompts=["sp"],
            few_shot_examples=[{"input": "  ", "output": "a1"}],
        )
    with pytest.raises(ValidationError, match="non-empty"):
        PromptVariants(
            system_prompts=["sp"],
            few_shot_examples=[{"input": "q1", "output": ""}],
        )


def test_prompt_variants_few_shot_accepts_well_formed_shots():
    pv = PromptVariants(
        system_prompts=["sp"],
        few_shot_examples=[
            {"input": "q1", "output": "a1"},
            {"input": "q2", "output": "a2"},
        ],
    )
    assert len(pv.few_shot_examples) == 2


# ---------------------------------------------------------------------------
# Reviewer P1 #10: cross-validation of MetaAxisSpace against PromptVariants.
# A user-supplied space whose bounds exceed the variant pool would silently
# IndexError mid-eval (system_prompt_variant) or short-slice the few-shot
# list (mis-recording few_shot_count in the artifact).
# ---------------------------------------------------------------------------


def test_validate_space_against_variants_accepts_consistent_pair():
    space = MetaAxisSpace(system_prompt_idx_max=1, few_shot_max=2)
    variants = PromptVariants(
        system_prompts=["sp-A", "sp-B"],
        few_shot_examples=[
            {"input": "q1", "output": "a1"},
            {"input": "q2", "output": "a2"},
        ],
    )
    validate_space_against_variants(space, variants)  # no raise


def test_validate_space_against_variants_rejects_idx_max_too_large():
    space = MetaAxisSpace(system_prompt_idx_max=5)
    variants = PromptVariants(
        system_prompts=["sp-A", "sp-B"],
    )
    with pytest.raises(ValueError, match="system_prompt_idx_max"):
        validate_space_against_variants(space, variants)


def test_validate_space_against_variants_rejects_few_shot_max_too_large():
    space = MetaAxisSpace(system_prompt_idx_max=0, few_shot_max=5)
    variants = PromptVariants(
        system_prompts=["sp-A"],
        few_shot_examples=[{"input": "q1", "output": "a1"}],
    )
    with pytest.raises(ValueError, match="few_shot_max"):
        validate_space_against_variants(space, variants)


def test_validate_space_against_variants_allows_few_shot_max_equal_to_pool():
    """Boundary case: few_shot_max == pool size means the search can pick
    every example, which is valid (slicing returns the full list)."""
    space = MetaAxisSpace(system_prompt_idx_max=0, few_shot_max=2)
    variants = PromptVariants(
        system_prompts=["sp"],
        few_shot_examples=[
            {"input": "q1", "output": "a1"},
            {"input": "q2", "output": "a2"},
        ],
    )
    validate_space_against_variants(space, variants)  # no raise
