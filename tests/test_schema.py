"""Schema validation tests - v1.0 domain types (and legacy shim aliases)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omegaprompt.domain.enums import OutputBudgetBucket, ReasoningProfile
from omegaprompt.domain.params import MetaAxisSpace, PromptVariants
from omegaprompt.domain.result import CalibrationArtifact
from omegaprompt.schema import (  # legacy shim aliases
    CalibrationOutcome,
    ParamVariants,
    PromptSpace,
)


def test_prompt_variants_rejects_empty_system_prompts():
    with pytest.raises(ValidationError):
        PromptVariants(system_prompts=[])


def test_prompt_variants_rejects_blank_system_prompt():
    with pytest.raises(ValidationError):
        PromptVariants(system_prompts=["", "real prompt"])


def test_prompt_variants_accepts_valid():
    v = PromptVariants(
        system_prompts=["a", "b"],
        few_shot_examples=[{"input": "i", "output": "o"}],
    )
    assert len(v.system_prompts) == 2
    assert v.few_shot_examples[0]["input"] == "i"


def test_legacy_param_variants_alias_resolves_to_new_type():
    assert ParamVariants is PromptVariants


def test_meta_axis_space_defaults_sane():
    s = MetaAxisSpace(system_prompt_idx_max=4)
    assert s.few_shot_min == 0
    assert s.few_shot_max == 3
    assert ReasoningProfile.STANDARD in s.reasoning_profiles
    assert OutputBudgetBucket.MEDIUM in s.output_budgets


def test_meta_axis_space_few_shot_max_must_ge_min():
    with pytest.raises(ValidationError):
        MetaAxisSpace(system_prompt_idx_max=2, few_shot_min=5, few_shot_max=2)


def test_meta_axis_space_rejects_empty_axis_lists():
    with pytest.raises(ValidationError):
        MetaAxisSpace(system_prompt_idx_max=2, reasoning_profiles=[])
    with pytest.raises(ValidationError):
        MetaAxisSpace(system_prompt_idx_max=2, output_budgets=[])


def test_legacy_prompt_space_alias_resolves_to_meta_axis_space():
    assert PromptSpace is MetaAxisSpace


def test_calibration_artifact_required_fields():
    with pytest.raises(ValidationError):
        CalibrationArtifact()


def test_calibration_artifact_happy_path():
    a = CalibrationArtifact(
        method="p1",
        unlock_k=3,
        best_params={"system_prompt_variant": 1},
        best_fitness=0.75,
        neutral_baseline_params={"system_prompt_variant": 0},
        calibrated_params={"system_prompt_variant": 1},
        neutral_fitness=0.50,
        calibrated_fitness=0.75,
        hard_gate_pass_rate=0.9,
        n_candidates_evaluated=50,
        total_api_calls=500,
    )
    assert a.walk_forward is None
    assert a.status == "OK"
    payload = a.model_dump_json()
    assert '"best_fitness":0.75' in payload
    assert a.calibrated_params["system_prompt_variant"] == 1


def test_calibration_artifact_pass_rate_clamped_to_unit():
    with pytest.raises(ValidationError):
        CalibrationArtifact(
            method="grid",
            unlock_k=1,
            best_params={},
            best_fitness=1.0,
            hard_gate_pass_rate=1.5,
            n_candidates_evaluated=0,
            total_api_calls=0,
        )


def test_legacy_calibration_outcome_alias_resolves_to_artifact():
    assert CalibrationOutcome is CalibrationArtifact
