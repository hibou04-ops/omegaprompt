"""Analytical preflight (mini-antemortem adapter).

Reads the run configuration and emits classifications over a fixed set
of calibration-specific trap patterns. No API calls; all reasoning is
deterministic given the config.

The full antemortem-cli ``--domain calibration`` integration is a separate
project that uses an LLM to reason over the trap list against richer
evidence (vendor docs, prior-run artifacts, model cards). The in-process
version below covers the highest-signal traps with deterministic rules so
the core pipeline can stand alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from omegaprompt.domain.dataset import Dataset
from omegaprompt.domain.judge import JudgeRubric
from omegaprompt.domain.params import PromptVariants
from omegaprompt.preflight.contracts import (
    AnalyticalFinding,
    PreflightSeverity,
)


@dataclass(frozen=True)
class TrapPattern:
    """One reusable calibration trap pattern."""

    id: str
    hypothesis: str


CALIBRATION_TRAPS: tuple[TrapPattern, ...] = (
    TrapPattern(
        id="self_agreement_bias",
        hypothesis=(
            "Target and judge share a vendor; judge's biases overlap with the "
            "target, flattering same-vendor responses."
        ),
    ),
    TrapPattern(
        id="small_sample_kc4_power",
        hypothesis=(
            "Dataset is small enough that Pearson KC-4 has no statistical power; "
            "correlation threshold becomes a random pass/fail."
        ),
    ),
    TrapPattern(
        id="variants_homogeneous",
        hypothesis=(
            "System prompt variants are too similar; sensitivity on the "
            "system_prompt_variant axis will be artificially low."
        ),
    ),
    TrapPattern(
        id="rubric_weight_concentration",
        hypothesis=(
            "A single rubric dimension carries the majority of the weight; "
            "judge noise on that one dimension dominates the fitness."
        ),
    ),
    TrapPattern(
        id="judge_budget_too_small",
        hypothesis=(
            "Judge output budget is SMALL but rubric has many dimensions + "
            "gates; judge response may be truncated before scoring all axes."
        ),
    ),
    TrapPattern(
        id="empty_reference_with_strict_rubric",
        hypothesis=(
            "Dataset items have no reference text while the rubric's "
            "dimension descriptions imply comparison to a ground truth."
        ),
    ),
    TrapPattern(
        id="no_held_out_slice",
        hypothesis=(
            "User did not pass --test; walk-forward cannot run; ship "
            "decision has no generalisation evidence."
        ),
    ),
)


def analytical_traps() -> tuple[TrapPattern, ...]:
    """Return the built-in calibration trap patterns."""
    return CALIBRATION_TRAPS


def _finding(
    trap: TrapPattern,
    *,
    label: str,
    severity: PreflightSeverity = PreflightSeverity.MEDIUM,
    note: str = "",
    remediation: str = "",
    cite: str | None = None,
) -> AnalyticalFinding:
    return AnalyticalFinding(
        trap_id=trap.id,
        label=label,
        hypothesis=trap.hypothesis,
        severity=severity,
        note=note,
        remediation=remediation,
        cite=cite,
    )


def _check_self_agreement(
    target_provider: str,
    target_model: str | None,
    judge_provider: str,
    judge_model: str | None,
) -> AnalyticalFinding:
    trap = next(t for t in CALIBRATION_TRAPS if t.id == "self_agreement_bias")
    if target_provider == judge_provider and target_model == judge_model:
        return _finding(
            trap,
            label="REAL",
            severity=PreflightSeverity.HIGH,
            note=(
                f"Target and judge are identical: {target_provider}/{target_model or 'default'}. "
                "Judge will share the target's distributional biases."
            ),
            remediation="Use a different vendor or stronger model for --judge-*.",
        )
    if target_provider == judge_provider:
        return _finding(
            trap,
            label="REAL",
            severity=PreflightSeverity.MEDIUM,
            note=f"Target and judge share vendor ({target_provider}); some bias overlap.",
            remediation="Consider a cross-vendor judge to break self-agreement bias.",
        )
    return _finding(
        trap,
        label="GHOST",
        severity=PreflightSeverity.LOW,
        note=f"Target on {target_provider}, judge on {judge_provider} - different vendors.",
    )


def _check_sample_power(train_size: int, test_size: int | None) -> AnalyticalFinding:
    trap = next(t for t in CALIBRATION_TRAPS if t.id == "small_sample_kc4_power")
    total = train_size + (test_size or 0)
    if test_size is None:
        # KC-4 won't run; separate trap handles that.
        return _finding(
            trap,
            label="GHOST",
            severity=PreflightSeverity.LOW,
            note="No --test slice provided; Pearson check will not execute.",
        )
    if test_size < 10:
        return _finding(
            trap,
            label="REAL",
            severity=PreflightSeverity.HIGH,
            note=(
                f"Test slice has {test_size} items. Pearson correlation at n={test_size} "
                "has weak statistical power; KC-4 pass/fail may be random."
            ),
            remediation=(
                "Expand test set to at least 20 items, or raise --min-kc4 adaptively "
                "(handled by AdaptationPlan)."
            ),
        )
    if total < 20:
        return _finding(
            trap,
            label="REAL",
            severity=PreflightSeverity.MEDIUM,
            note=f"Total dataset is {total} items; noise absorption limited.",
            remediation="Larger datasets yield more reliable calibration gradients.",
        )
    return _finding(
        trap,
        label="GHOST",
        severity=PreflightSeverity.LOW,
        note=f"Dataset size {total} adequate for Pearson power.",
    )


def _check_variants_homogeneity(variants: PromptVariants) -> AnalyticalFinding:
    trap = next(t for t in CALIBRATION_TRAPS if t.id == "variants_homogeneous")
    prompts = variants.system_prompts
    if len(prompts) < 2:
        return _finding(
            trap,
            label="REAL",
            severity=PreflightSeverity.MEDIUM,
            note="Only one system-prompt variant; axis contributes zero search signal.",
            remediation="Provide at least 3 genuinely distinct system prompts.",
        )
    lengths = [len(p) for p in prompts]
    if max(lengths) - min(lengths) < 20 and len(prompts) <= 3:
        return _finding(
            trap,
            label="NEW",
            severity=PreflightSeverity.MEDIUM,
            note=(
                f"All {len(prompts)} system prompts have near-identical length "
                f"({min(lengths)}-{max(lengths)} chars); they may be too similar to "
                "produce meaningful sensitivity."
            ),
            remediation=(
                "Author variants that differ in role framing, not just wording; "
                "vary length by at least 2x where possible."
            ),
        )
    return _finding(
        trap,
        label="GHOST",
        severity=PreflightSeverity.LOW,
        note=(
            f"System-prompt variants span {min(lengths)}-{max(lengths)} chars; "
            "sufficient diversity expected."
        ),
    )


def _check_rubric_concentration(rubric: JudgeRubric) -> AnalyticalFinding:
    trap = next(t for t in CALIBRATION_TRAPS if t.id == "rubric_weight_concentration")
    weights = rubric.normalized_weights()
    if not weights:
        return _finding(trap, label="UNRESOLVED")
    max_w = max(weights.values())
    max_name = max(weights, key=weights.get)  # type: ignore[arg-type]
    if max_w > 0.7:
        return _finding(
            trap,
            label="REAL",
            severity=PreflightSeverity.MEDIUM,
            note=(
                f"Dimension '{max_name}' carries {max_w:.0%} of the rubric weight; "
                "judge noise on that single dimension will dominate fitness."
            ),
            remediation=(
                "Rebalance rubric so no single dimension exceeds ~50% weight, "
                "or explicitly declare this concentration is intentional."
            ),
        )
    return _finding(
        trap,
        label="GHOST",
        severity=PreflightSeverity.LOW,
        note=f"Max dimension weight is {max_w:.0%}; no single-dim dominance.",
    )


def _check_judge_budget(rubric: JudgeRubric, judge_output_budget: str) -> AnalyticalFinding:
    trap = next(t for t in CALIBRATION_TRAPS if t.id == "judge_budget_too_small")
    n_dims = len(rubric.dimensions)
    n_gates = len([g for g in rubric.hard_gates if g.evaluator == "judge"])
    total_axes = n_dims + n_gates
    if judge_output_budget == "small" and total_axes > 5:
        return _finding(
            trap,
            label="REAL",
            severity=PreflightSeverity.MEDIUM,
            note=(
                f"Judge budget SMALL (1024 tokens) vs {total_axes} rubric axes "
                f"({n_dims} dims + {n_gates} gates). JudgeResult response may be truncated."
            ),
            remediation="Raise LLMJudge output_budget to MEDIUM or reduce rubric surface.",
        )
    return _finding(
        trap,
        label="GHOST",
        severity=PreflightSeverity.LOW,
        note=f"Judge budget {judge_output_budget} adequate for {total_axes} axes.",
    )


def _check_empty_reference(dataset: Dataset) -> AnalyticalFinding:
    trap = next(t for t in CALIBRATION_TRAPS if t.id == "empty_reference_with_strict_rubric")
    has_ref = sum(1 for it in dataset.items if it.reference)
    total = len(dataset.items)
    if has_ref == 0 and total > 0:
        return _finding(
            trap,
            label="NEW",
            severity=PreflightSeverity.LOW,
            note=(
                "No dataset item has a reference field; judge scores purely by rubric "
                "without ground-truth anchor."
            ),
            remediation=(
                "If your rubric's descriptions imply comparison to a reference, add "
                "reference fields or reword the rubric to be self-contained."
            ),
        )
    return _finding(
        trap,
        label="GHOST",
        severity=PreflightSeverity.LOW,
        note=f"{has_ref}/{total} items carry reference text.",
    )


def _check_no_held_out(has_test_slice: bool) -> AnalyticalFinding:
    trap = next(t for t in CALIBRATION_TRAPS if t.id == "no_held_out_slice")
    if not has_test_slice:
        return _finding(
            trap,
            label="REAL",
            severity=PreflightSeverity.HIGH,
            note=(
                "No --test slice was provided. Walk-forward validation will not run "
                "and the artifact's ship recommendation will be HOLD regardless of "
                "training fitness."
            ),
            remediation="Provide --test <held_out.jsonl> for walk-forward validation.",
        )
    return _finding(
        trap,
        label="GHOST",
        severity=PreflightSeverity.LOW,
        note="Held-out slice provided; walk-forward will run.",
    )


def analytical_preflight(
    *,
    target_provider: str,
    target_model: str | None,
    judge_provider: str,
    judge_model: str | None,
    train_dataset: Dataset,
    test_dataset: Dataset | None,
    rubric: JudgeRubric,
    variants: PromptVariants,
    judge_output_budget: str = "small",
) -> list[AnalyticalFinding]:
    """Run analytical preflight checks and return one finding per trap.

    All checks are deterministic given the inputs. The ordering of the
    returned list is stable (same order as :data:`CALIBRATION_TRAPS`).
    """
    findings: list[AnalyticalFinding] = [
        _check_self_agreement(target_provider, target_model, judge_provider, judge_model),
        _check_sample_power(
            train_size=len(train_dataset),
            test_size=len(test_dataset) if test_dataset is not None else None,
        ),
        _check_variants_homogeneity(variants),
        _check_rubric_concentration(rubric),
        _check_judge_budget(rubric, judge_output_budget),
        _check_empty_reference(train_dataset),
        _check_no_held_out(has_test_slice=test_dataset is not None),
    ]
    return findings
