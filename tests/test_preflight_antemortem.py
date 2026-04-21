"""Analytical preflight tests - trap classifications over a range of configs."""

from __future__ import annotations

from omegaprompt.domain.dataset import Dataset, DatasetItem
from omegaprompt.domain.judge import Dimension, HardGate, JudgeRubric
from omegaprompt.domain.params import PromptVariants
from omegaprompt.preflight.contracts import PreflightSeverity
from omegaprompt.preflight.mini_antemortem import analytical_preflight, analytical_traps


def _rubric(dim_weights=None, gates=1) -> JudgeRubric:
    if dim_weights is None:
        dim_weights = {"accuracy": 0.7, "clarity": 0.3}
    return JudgeRubric(
        dimensions=[
            Dimension(name=name, description=f"{name} description", weight=w)
            for name, w in dim_weights.items()
        ],
        hard_gates=[
            HardGate(name=f"g{i}", description=f"gate {i}", evaluator="judge")
            for i in range(gates)
        ],
    )


def _variants(n_prompts=3, lens=(100, 200, 500)) -> PromptVariants:
    prompts = ["X" * lens[i % len(lens)] for i in range(n_prompts)]
    return PromptVariants(
        system_prompts=prompts,
        few_shot_examples=[{"input": "1+1", "output": "2"}],
    )


def _dataset(n=5, with_ref=False) -> Dataset:
    items = [
        DatasetItem(
            id=f"t{i}",
            input=f"task {i}",
            reference=f"ref {i}" if with_ref else None,
        )
        for i in range(n)
    ]
    return Dataset(items=items)


def _by_trap(findings, trap_id):
    return next(f for f in findings if f.trap_id == trap_id)


def test_trap_registry_contains_all_expected():
    trap_ids = {t.id for t in analytical_traps()}
    assert trap_ids == {
        "self_agreement_bias",
        "small_sample_kc4_power",
        "variants_homogeneous",
        "rubric_weight_concentration",
        "judge_budget_too_small",
        "empty_reference_with_strict_rubric",
        "no_held_out_slice",
    }


def test_self_agreement_identical_is_real_high():
    findings = analytical_preflight(
        target_provider="openai",
        target_model="gpt-4o",
        judge_provider="openai",
        judge_model="gpt-4o",
        train_dataset=_dataset(n=20, with_ref=True),
        test_dataset=_dataset(n=15, with_ref=True),
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "self_agreement_bias")
    assert f.label == "REAL"
    assert f.severity == PreflightSeverity.HIGH


def test_self_agreement_same_vendor_different_model_is_real_medium():
    findings = analytical_preflight(
        target_provider="openai",
        target_model="gpt-4o",
        judge_provider="openai",
        judge_model="gpt-4o-mini",
        train_dataset=_dataset(n=20, with_ref=True),
        test_dataset=_dataset(n=15, with_ref=True),
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "self_agreement_bias")
    assert f.label == "REAL"
    assert f.severity == PreflightSeverity.MEDIUM


def test_self_agreement_cross_vendor_is_ghost():
    findings = analytical_preflight(
        target_provider="openai",
        target_model="gpt-4o",
        judge_provider="anthropic",
        judge_model="claude-opus-4-7",
        train_dataset=_dataset(n=20, with_ref=True),
        test_dataset=_dataset(n=15, with_ref=True),
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "self_agreement_bias")
    assert f.label == "GHOST"


def test_small_sample_power_flags_small_test_slice():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=5),
        test_dataset=_dataset(n=5),
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "small_sample_kc4_power")
    assert f.label == "REAL"
    assert f.severity == PreflightSeverity.HIGH


def test_small_sample_power_ghost_when_no_test_slice():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=5),
        test_dataset=None,
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "small_sample_kc4_power")
    assert f.label == "GHOST"


def test_variants_homogeneous_flags_uniform_length_variants():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=20),
        test_dataset=_dataset(n=15),
        rubric=_rubric(),
        variants=PromptVariants(
            system_prompts=["You are an assistant.", "You are a helper."],
            few_shot_examples=[],
        ),
    )
    f = _by_trap(findings, "variants_homogeneous")
    assert f.label == "NEW"


def test_variants_single_prompt_is_real():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=20),
        test_dataset=_dataset(n=15),
        rubric=_rubric(),
        variants=PromptVariants(system_prompts=["only one"], few_shot_examples=[]),
    )
    f = _by_trap(findings, "variants_homogeneous")
    assert f.label == "REAL"


def test_rubric_weight_concentration_flags_over_70():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=20),
        test_dataset=_dataset(n=15),
        rubric=_rubric(dim_weights={"accuracy": 0.9, "clarity": 0.1}),
        variants=_variants(),
    )
    f = _by_trap(findings, "rubric_weight_concentration")
    assert f.label == "REAL"


def test_rubric_weight_concentration_ghost_on_balanced():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=20),
        test_dataset=_dataset(n=15),
        rubric=_rubric(dim_weights={"accuracy": 0.5, "clarity": 0.5}),
        variants=_variants(),
    )
    f = _by_trap(findings, "rubric_weight_concentration")
    assert f.label == "GHOST"


def test_judge_budget_too_small_flags_many_axes_on_small_budget():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=20),
        test_dataset=_dataset(n=15),
        rubric=_rubric(
            dim_weights={f"d{i}": 1.0 / 6 for i in range(6)},
            gates=2,
        ),
        variants=_variants(),
        judge_output_budget="small",
    )
    f = _by_trap(findings, "judge_budget_too_small")
    assert f.label == "REAL"


def test_judge_budget_ghost_on_medium_budget():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=20),
        test_dataset=_dataset(n=15),
        rubric=_rubric(),
        variants=_variants(),
        judge_output_budget="medium",
    )
    f = _by_trap(findings, "judge_budget_too_small")
    assert f.label == "GHOST"


def test_empty_reference_flagged_new_when_no_refs():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=20, with_ref=False),
        test_dataset=_dataset(n=15, with_ref=False),
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "empty_reference_with_strict_rubric")
    assert f.label == "NEW"


def test_empty_reference_ghost_when_refs_present():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=20, with_ref=True),
        test_dataset=_dataset(n=15, with_ref=True),
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "empty_reference_with_strict_rubric")
    assert f.label == "GHOST"


def test_no_held_out_slice_real_when_missing():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=20),
        test_dataset=None,
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "no_held_out_slice")
    assert f.label == "REAL"
    assert f.severity == PreflightSeverity.HIGH


def test_no_held_out_slice_ghost_when_provided():
    findings = analytical_preflight(
        target_provider="anthropic",
        target_model="claude",
        judge_provider="openai",
        judge_model="gpt",
        train_dataset=_dataset(n=20),
        test_dataset=_dataset(n=15),
        rubric=_rubric(),
        variants=_variants(),
    )
    f = _by_trap(findings, "no_held_out_slice")
    assert f.label == "GHOST"
