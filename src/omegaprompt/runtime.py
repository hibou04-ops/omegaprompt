"""High-level entrypoints for omegaprompt — the agent-callable surface.

These four functions wrap the calibration pipeline as one-call operations,
coercing inputs (paths, dicts, or pre-built objects) to the canonical
domain types and returning structured results.

Designed for MCP wrappers, agent integrations, and Python callers who
want a single function rather than the composable underlying API.

Tier 1 (this module): calibrate, evaluate, report, diff.
Tier 2 (forthcoming):  measure_sensitivity, judge, preflight, classify_traps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

from pydantic import BaseModel, Field

from omegaprompt.core import (
    assess_run_risk,
    evaluate_walk_forward,
    load_artifact,
    policy_for,
    relaxed_safeguards_for,
    save_artifact,
)
from omegaprompt.domain import (
    CalibrationArtifact,
    Dataset,
    DatasetItem,
    EvalResult,
    ExecutionProfile,
    JudgeResult,
    JudgeRubric,
    MetaAxisSpace,
    PromptVariants,
    ResolvedPromptParams,
    ShipRecommendation,
)
from omegaprompt.judges import EnsembleJudge, Judge, LLMJudge, RuleJudge
from omegaprompt.judges.rule_judge import default_no_refusal, default_non_empty
from omegaprompt.preflight.contracts import (
    AnalyticalFinding,
    PreflightReport,
    PreflightStatus,
)
from omegaprompt.providers import (
    LLMProvider,
    make_provider,
    provider_capabilities,
    quality_per_cost,
    quality_per_latency,
)
from omegaprompt.reporting import render_markdown
from omegaprompt.targets import PromptTarget


# ----------------------------------------------------------------------
# Public Pydantic types — schema-derivable for MCP tool definitions
# ----------------------------------------------------------------------


class ProviderSpec(BaseModel):
    """Compact provider configuration. Coerced to LLMProvider internally.

    Use this when you don't already have a built LLMProvider instance.
    """

    name: str
    model: str | None = None
    base_url: str | None = None

    model_config = {"extra": "forbid"}


class CalibrateTuning(BaseModel):
    """Low-frequency tuning knobs for calibrate(). Defaults from execution profile.

    Agents typically leave this as None and accept profile defaults; callers
    who need fine control over walk-forward gates, axis space, or search
    method can populate it.
    """

    method: str = "p1"
    unlock_k: int = 3
    space: MetaAxisSpace | None = None
    max_gap: float | None = None
    min_kc4: float | None = None
    profile: ExecutionProfile = ExecutionProfile.GUARDED

    model_config = {"extra": "forbid"}


class ArtifactDiff(BaseModel):
    """Structured comparison of two CalibrationArtifacts."""

    baseline_status: str
    candidate_status: str
    fitness_delta: float
    neutral_fitness_delta: float
    walk_forward_test_delta: float | None = None
    walk_forward_passed_change: tuple[bool, bool] | None = None
    hard_gate_pass_rate_delta: float
    quality_per_cost_delta: float
    quality_per_latency_delta: float
    regressed: bool
    regression_reasons: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class SensitivityTuning(BaseModel):
    """Low-frequency tuning knobs for measure_sensitivity()."""

    space: MetaAxisSpace | None = None
    profile: ExecutionProfile = ExecutionProfile.GUARDED

    model_config = {"extra": "forbid"}


class SensitivityResult(BaseModel):
    """Per-axis sensitivity ranking from a stress probe.

    Each ``rows`` entry has ``axis``, ``normalized_stress``, ``raw_stress``,
    and ``rank`` (0 = most sensitive). Ordered by descending sensitivity.
    """

    rows: list[dict] = Field(default_factory=list)
    baseline_fitness: float
    n_probes: int

    model_config = {"extra": "forbid"}


# ----------------------------------------------------------------------
# Input coercion helpers — accept multiple input shapes per entrypoint
# ----------------------------------------------------------------------


def _resolve_provider(p: object) -> LLMProvider:
    if isinstance(p, str):
        return make_provider(p)
    if isinstance(p, ProviderSpec):
        return make_provider(p.name, model=p.model, base_url=p.base_url)
    if isinstance(p, dict):
        return make_provider(
            p["name"], model=p.get("model"), base_url=p.get("base_url")
        )
    return p  # already an LLMProvider


def _resolve_dataset(d: object) -> Dataset:
    if isinstance(d, Dataset):
        return d
    if isinstance(d, (str, Path)):
        return Dataset.from_jsonl(Path(d))
    raise TypeError(f"Unsupported dataset input: {type(d).__name__}")


def _resolve_rubric(r: object) -> JudgeRubric:
    if isinstance(r, JudgeRubric):
        return r
    if isinstance(r, dict):
        return JudgeRubric.model_validate(r)
    if isinstance(r, (str, Path)):
        return JudgeRubric.from_json(Path(r))
    raise TypeError(f"Unsupported rubric input: {type(r).__name__}")


def _resolve_variants(v: object) -> PromptVariants:
    if isinstance(v, PromptVariants):
        return v
    if isinstance(v, dict):
        return PromptVariants.model_validate(v)
    if isinstance(v, (str, Path)):
        return PromptVariants.model_validate_json(
            Path(v).read_text(encoding="utf-8")
        )
    raise TypeError(f"Unsupported variants input: {type(v).__name__}")


def _resolve_artifact(a: object) -> CalibrationArtifact:
    if isinstance(a, CalibrationArtifact):
        return a
    if isinstance(a, (str, Path)):
        return load_artifact(Path(a))
    raise TypeError(f"Unsupported artifact input: {type(a).__name__}")


def _resolve_params(p: object) -> dict:
    if isinstance(p, ResolvedPromptParams):
        return p.model_dump()
    if isinstance(p, CalibrationArtifact):
        return p.calibrated_params.model_dump()
    if isinstance(p, dict):
        return p
    raise TypeError(f"Unsupported params input: {type(p).__name__}")


# ----------------------------------------------------------------------
# Tier 1 entrypoints
# ----------------------------------------------------------------------


def calibrate(
    train: Dataset | Path | str,
    *,
    rubric: JudgeRubric | Path | str | dict,
    variants: PromptVariants | Path | str | dict,
    target: LLMProvider | ProviderSpec | str | dict,
    judge: LLMProvider | ProviderSpec | str | dict | None = None,
    test: Dataset | Path | str | None = None,
    output: Path | str | None = None,
    tuning: CalibrateTuning | None = None,
) -> CalibrationArtifact:
    """End-to-end calibration: sensitivity → grid → walk-forward → artifact.

    Most expensive entrypoint (dozens to hundreds of LLM calls). Use case:
    agent or CI bot validating a prompt configuration before ship.

    If ``judge`` is ``None``, target is reused as judge (simplest setup but
    enables self-agreement bias — provide a different vendor's judge for a
    stronger signal).

    If ``output`` is set, the artifact is also written to disk as JSON; the
    function still returns the in-memory artifact.
    """
    try:
        from omega_lock import P1Config, run_p1
    except ImportError as exc:
        raise ImportError(
            "omegaprompt.calibrate requires omega-lock. "
            "Install with `pip install omega-lock`."
        ) from exc

    tuning = tuning or CalibrateTuning()
    profile = (
        tuning.profile
        if isinstance(tuning.profile, ExecutionProfile)
        else ExecutionProfile(tuning.profile)
    )
    policy = policy_for(profile)
    resolved_max_gap = (
        tuning.max_gap if tuning.max_gap is not None else policy.default_max_gap
    )
    resolved_min_kc4 = (
        tuning.min_kc4 if tuning.min_kc4 is not None else policy.default_min_kc4
    )

    train_ds = _resolve_dataset(train)
    test_ds = _resolve_dataset(test) if test is not None else None
    rubric_obj = _resolve_rubric(rubric)
    variants_obj = _resolve_variants(variants)
    target_provider = _resolve_provider(target)
    judge_provider = (
        _resolve_provider(judge) if judge is not None else target_provider
    )

    judge_obj = LLMJudge(provider=judge_provider, execution_profile=profile)
    train_target = PromptTarget(
        target_provider=target_provider,
        judge=judge_obj,
        dataset=train_ds,
        rubric=rubric_obj,
        variants=variants_obj,
        space=tuning.space,
        execution_profile=profile,
    )
    test_target = (
        PromptTarget(
            target_provider=target_provider,
            judge=judge_obj,
            dataset=test_ds,
            rubric=rubric_obj,
            variants=variants_obj,
            space=tuning.space,
            execution_profile=profile,
        )
        if test_ds is not None
        else None
    )

    neutral_result = train_target.evaluate(train_target.neutral_params())
    config = P1Config(unlock_k=tuning.unlock_k)
    result = run_p1(
        train_target=train_target, test_target=test_target, config=config
    )

    grid_best = getattr(result, "grid_best", None) or {}
    best_unlocked: dict = (
        grid_best.get("unlocked", {}) if isinstance(grid_best, dict) else {}
    )
    best_candidate = {**train_target.neutral_params(), **best_unlocked}
    best_train_eval = train_target.evaluate(best_candidate)
    best_guarded_eval = train_target.best_guarded_eval()

    walk_forward = None
    test_eval = None
    status = "OK"
    rationale = "passed"
    if test_target is not None:
        test_eval = test_target.evaluate(best_candidate)
        walk_forward = evaluate_walk_forward(
            best_train_eval.fitness,
            test_eval.fitness,
            per_item_train=_per_item_scores(best_train_eval),
            per_item_test=_per_item_scores(test_eval),
            max_gap=resolved_max_gap,
            min_kc4=resolved_min_kc4,
        )
        if not walk_forward.passed:
            status = "FAIL_KC4_GATE"
            rationale = (
                f"train={best_train_eval.fitness:.3f} "
                f"test={test_eval.fitness:.3f} "
                f"gap={walk_forward.generalization_gap:.2%} "
                f"kc4={walk_forward.kc4_correlation}"
            )

    sensitivity_rows = _sensitivity_rows(getattr(result, "stress_results", None))
    degraded_capabilities = _dedupe_events(
        [
            *neutral_result.degraded_capabilities,
            *best_train_eval.degraded_capabilities,
            *(test_eval.degraded_capabilities if test_eval is not None else []),
        ]
    )
    target_caps = provider_capabilities(target_provider)
    judge_caps = provider_capabilities(judge_provider)
    boundary_warnings, within_guarded, ship_recommendation = assess_run_risk(
        profile=profile,
        target_capabilities=target_caps,
        judge_capabilities=judge_caps,
        degraded_capabilities=degraded_capabilities,
        has_walk_forward=test_target is not None,
        walk_forward_passed=None if walk_forward is None else walk_forward.passed,
    )

    calibrated_fitness = best_train_eval.fitness
    neutral_fitness = neutral_result.fitness
    uplift_absolute = calibrated_fitness - neutral_fitness
    uplift_percent = (
        (uplift_absolute / neutral_fitness * 100.0) if neutral_fitness else 0.0
    )

    additional_uplift = 0.0
    if not best_train_eval.within_guarded_boundaries:
        if best_guarded_eval is not None:
            additional_uplift = calibrated_fitness - best_guarded_eval.fitness
        else:
            additional_uplift = uplift_absolute

    if ship_recommendation.value == "block" and status == "OK":
        status = "FAIL_HARD_GATES"
        rationale = "structural risk exceeded the current profile boundary"

    artifact = CalibrationArtifact(
        method=tuning.method,
        unlock_k=tuning.unlock_k,
        selected_profile=profile,
        neutral_baseline_params=neutral_result.resolved_params,
        calibrated_params=best_train_eval.resolved_params,
        neutral_fitness=neutral_fitness,
        calibrated_fitness=calibrated_fitness,
        uplift_absolute=uplift_absolute,
        uplift_percent=uplift_percent,
        quality_per_cost_neutral=quality_per_cost(
            neutral_fitness, neutral_result.estimated_cost_units
        ),
        quality_per_cost_best=quality_per_cost(
            calibrated_fitness, best_train_eval.estimated_cost_units
        ),
        quality_per_latency_neutral=quality_per_latency(
            neutral_fitness, neutral_result.latency_ms
        ),
        quality_per_latency_best=quality_per_latency(
            calibrated_fitness, best_train_eval.latency_ms
        ),
        boundary_warnings=boundary_warnings,
        degraded_capabilities=degraded_capabilities,
        ship_recommendation=ship_recommendation,
        stayed_within_guarded_boundaries=within_guarded
        and best_train_eval.within_guarded_boundaries,
        additional_uplift_from_boundary_crossing=additional_uplift,
        relaxed_safeguards=relaxed_safeguards_for(profile),
        guarded_boundary_crossed=not (
            within_guarded and best_train_eval.within_guarded_boundaries
        ),
        best_params=best_train_eval.resolved_params,
        best_fitness=calibrated_fitness,
        walk_forward=walk_forward,
        hard_gate_pass_rate=best_train_eval.hard_gate_pass_rate,
        sensitivity_ranking=sensitivity_rows,
        n_candidates_evaluated=train_target.unique_param_count(),
        total_api_calls=train_target.total_api_calls
        + (test_target.total_api_calls if test_target else 0),
        usage_summary=_merge_usage(
            train_target.last_usage,
            test_target.last_usage if test_target is not None else {},
        ),
        latency_summary_ms={
            "neutral_train": neutral_result.latency_ms,
            "calibrated_train": best_train_eval.latency_ms,
            "calibrated_test": test_eval.latency_ms if test_eval is not None else 0.0,
        },
        target_provider=target_provider.name,
        target_model=target_provider.model,
        judge_provider=judge_provider.name,
        judge_model=judge_provider.model,
        target_capabilities=target_caps,
        judge_capabilities=judge_caps,
        status=status,
        rationale=rationale,
    )

    if output is not None:
        save_artifact(artifact, Path(output))

    return artifact


def evaluate(
    dataset: Dataset | Path | str,
    *,
    rubric: JudgeRubric | Path | str | dict,
    variants: PromptVariants | Path | str | dict,
    params: ResolvedPromptParams | dict | CalibrationArtifact,
    target: LLMProvider | ProviderSpec | str | dict,
    judge: LLMProvider | ProviderSpec | str | dict | None = None,
    profile: ExecutionProfile | str = ExecutionProfile.GUARDED,
) -> EvalResult:
    """Evaluate a fixed prompt configuration against a dataset (no search).

    Use case: agent re-scoring a previously calibrated config on a new dataset
    (regression check) or scoring a fixed candidate before deciding whether
    full calibration is worth the cost.

    ``params`` accepts a CalibrationArtifact directly — its ``calibrated_params``
    is extracted automatically, so the common "evaluate the last best on the
    new test set" flow is one call.
    """
    profile_obj = (
        profile
        if isinstance(profile, ExecutionProfile)
        else ExecutionProfile(profile)
    )
    dataset_obj = _resolve_dataset(dataset)
    rubric_obj = _resolve_rubric(rubric)
    variants_obj = _resolve_variants(variants)
    target_provider = _resolve_provider(target)
    judge_provider = (
        _resolve_provider(judge) if judge is not None else target_provider
    )
    params_dict = _resolve_params(params)

    judge_obj = LLMJudge(provider=judge_provider, execution_profile=profile_obj)
    target_obj = PromptTarget(
        target_provider=target_provider,
        judge=judge_obj,
        dataset=dataset_obj,
        rubric=rubric_obj,
        variants=variants_obj,
        execution_profile=profile_obj,
    )
    return target_obj.evaluate(params_dict)


def report(artifact: CalibrationArtifact | Path | str) -> str:
    """Render a CalibrationArtifact as Markdown.

    Use case: produce a human-readable summary for PR descriptions, CI step
    output, or chat messages. Pure rendering, zero LLM calls.
    """
    return render_markdown(_resolve_artifact(artifact))


def diff(
    baseline: CalibrationArtifact | Path | str,
    candidate: CalibrationArtifact | Path | str,
    *,
    format: Literal["json", "markdown"] = "json",
) -> Union[ArtifactDiff, str]:
    """Compare two CalibrationArtifacts.

    Returns ``ArtifactDiff`` (format='json', default) or a markdown string
    (format='markdown'). Use case: CI regression detection between an old
    canonical artifact and a new candidate produced from edited variants.
    """
    old = _resolve_artifact(baseline)
    new = _resolve_artifact(candidate)

    fitness_delta = new.calibrated_fitness - old.calibrated_fitness
    neutral_fitness_delta = new.neutral_fitness - old.neutral_fitness

    walk_forward_test_delta: float | None = None
    walk_forward_passed_change: tuple[bool, bool] | None = None
    if old.walk_forward is not None and new.walk_forward is not None:
        walk_forward_test_delta = (
            new.walk_forward.test_fitness - old.walk_forward.test_fitness
        )
        walk_forward_passed_change = (
            old.walk_forward.passed,
            new.walk_forward.passed,
        )

    hard_gate_delta = new.hard_gate_pass_rate - old.hard_gate_pass_rate
    qpc_delta = new.quality_per_cost_best - old.quality_per_cost_best
    qpl_delta = new.quality_per_latency_best - old.quality_per_latency_best

    regression_reasons: list[str] = []
    # Status / ship_recommendation are first-class regression conditions:
    # a candidate that fails its own ship gate must not be marked clean
    # just because its raw metrics improved. Without these checks, a
    # candidate with status=FAIL_KC4_GATE but better fitness would slip
    # through CI as "no regression."
    if new.status != "OK":
        regression_reasons.append(f"candidate status is not OK: {new.status}")
    if new.ship_recommendation in (
        ShipRecommendation.BLOCK,
        ShipRecommendation.HOLD,
    ):
        regression_reasons.append(
            f"candidate ship_recommendation={new.ship_recommendation.value}"
        )
    if fitness_delta < 0:
        regression_reasons.append(
            f"calibrated_fitness regressed: {fitness_delta:+.4f}"
        )
    if hard_gate_delta < 0:
        regression_reasons.append(
            f"hard_gate_pass_rate regressed: {hard_gate_delta:+.1%}"
        )
    if qpc_delta < 0:
        regression_reasons.append("quality_per_cost regressed")
    if qpl_delta < 0:
        regression_reasons.append("quality_per_latency regressed")
    if walk_forward_test_delta is not None and walk_forward_test_delta < 0:
        regression_reasons.append(
            f"walk_forward test_fitness regressed: {walk_forward_test_delta:+.4f}"
        )
    if walk_forward_passed_change == (True, False):
        regression_reasons.append("walk_forward passed -> failed")
    if (
        old.stayed_within_guarded_boundaries
        and not new.stayed_within_guarded_boundaries
    ):
        regression_reasons.append(
            "stayed_within_guarded_boundaries: True -> False"
        )

    diff_obj = ArtifactDiff(
        baseline_status=old.status,
        candidate_status=new.status,
        fitness_delta=fitness_delta,
        neutral_fitness_delta=neutral_fitness_delta,
        walk_forward_test_delta=walk_forward_test_delta,
        walk_forward_passed_change=walk_forward_passed_change,
        hard_gate_pass_rate_delta=hard_gate_delta,
        quality_per_cost_delta=qpc_delta,
        quality_per_latency_delta=qpl_delta,
        regressed=bool(regression_reasons),
        regression_reasons=regression_reasons,
    )

    if format == "json":
        return diff_obj

    lines = ["# omegaprompt diff", ""]
    lines.append(
        f"baseline: status=`{old.status}` calibrated={old.calibrated_fitness:.4f}"
    )
    lines.append(
        f"candidate: status=`{new.status}` calibrated={new.calibrated_fitness:.4f}"
    )
    lines.append("")
    lines.append(
        f"- calibrated_fitness: {old.calibrated_fitness:.4f} -> "
        f"{new.calibrated_fitness:.4f} ({fitness_delta:+.4f})"
    )
    lines.append(
        f"- neutral_fitness: {old.neutral_fitness:.4f} -> "
        f"{new.neutral_fitness:.4f} ({neutral_fitness_delta:+.4f})"
    )
    lines.append(
        f"- hard_gate_pass_rate: {old.hard_gate_pass_rate:.1%} -> "
        f"{new.hard_gate_pass_rate:.1%}"
    )
    if walk_forward_test_delta is not None and old.walk_forward and new.walk_forward:
        lines.append(
            f"- walk_forward test_fitness: {old.walk_forward.test_fitness:.4f} -> "
            f"{new.walk_forward.test_fitness:.4f} ({walk_forward_test_delta:+.4f})"
        )
    if diff_obj.regressed:
        lines.append("")
        lines.append("## REGRESSION")
        for r in regression_reasons:
            lines.append(f"- {r}")
    else:
        lines.append("")
        lines.append("## OK")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Tier 2 entrypoints
# ----------------------------------------------------------------------


def measure_sensitivity(
    dataset: Dataset | Path | str,
    *,
    rubric: JudgeRubric | Path | str | dict,
    variants: PromptVariants | Path | str | dict,
    target: LLMProvider | ProviderSpec | str | dict,
    judge: LLMProvider | ProviderSpec | str | dict | None = None,
    tuning: SensitivityTuning | None = None,
) -> SensitivityResult:
    """Cheap axis-ranking probe — no grid search, no walk-forward.

    Use case: agent or developer deciding whether full calibration is
    worth the cost. If no axis carries signal (all near-zero stress),
    skip calibrate. If 2-3 axes dominate, calibrate is cheap and worth
    running.

    Roughly costs ``axes × probe_size`` LLM calls (typically 6-30 calls
    against a small dataset slice).
    """
    try:
        from omega_lock import measure_stress
    except ImportError as exc:
        raise ImportError(
            "omegaprompt.measure_sensitivity requires omega-lock. "
            "Install with `pip install omega-lock`."
        ) from exc

    tuning = tuning or SensitivityTuning()
    profile = (
        tuning.profile
        if isinstance(tuning.profile, ExecutionProfile)
        else ExecutionProfile(tuning.profile)
    )

    dataset_obj = _resolve_dataset(dataset)
    rubric_obj = _resolve_rubric(rubric)
    variants_obj = _resolve_variants(variants)
    target_provider = _resolve_provider(target)
    judge_provider = (
        _resolve_provider(judge) if judge is not None else target_provider
    )

    judge_obj = LLMJudge(provider=judge_provider, execution_profile=profile)
    target_obj = PromptTarget(
        target_provider=target_provider,
        judge=judge_obj,
        dataset=dataset_obj,
        rubric=rubric_obj,
        variants=variants_obj,
        space=tuning.space,
        execution_profile=profile,
    )

    baseline_params = target_obj.neutral_params()
    baseline_result = target_obj.evaluate(baseline_params)
    stress_results = measure_stress(target_obj, baseline_params, baseline_result)

    rows: list[dict] = []
    ranked = sorted(
        stress_results,
        key=lambda s: float(getattr(s, "normalized_stress", 0.0) or 0.0),
        reverse=True,
    )
    for rank, s in enumerate(ranked):
        rows.append(
            {
                "axis": getattr(s, "axis", None) or getattr(s, "name", None),
                "normalized_stress": getattr(s, "normalized_stress", None),
                "raw_stress": getattr(s, "raw_stress", None),
                "rank": rank,
            }
        )

    return SensitivityResult(
        rows=rows,
        baseline_fitness=baseline_result.fitness,
        n_probes=len(stress_results),
    )


def grade(
    *,
    rubric: JudgeRubric | Path | str | dict,
    item: DatasetItem | dict,
    response: str,
    provider: LLMProvider | ProviderSpec | str | dict,
    strategy: Literal["rule", "llm", "ensemble"] = "ensemble",
) -> JudgeResult:
    """Score a single response against a rubric.

    Use case: agent self-grading its own output before returning it; or
    spot-checking a candidate response without running a full evaluation.

    ``strategy`` selects the judge: ``rule`` for zero-API deterministic
    gate checks, ``llm`` for LLM scoring, ``ensemble`` (default) runs
    rule first and short-circuits if hard gates fail.

    Named ``grade`` (not ``judge``) to avoid shadowing the legacy
    ``omegaprompt.judge`` shim module at package level.
    """
    rubric_obj = _resolve_rubric(rubric)
    if isinstance(item, DatasetItem):
        item_obj = item
    elif isinstance(item, dict):
        item_obj = DatasetItem.model_validate(item)
    else:
        raise TypeError(f"Unsupported item input: {type(item).__name__}")

    default_checks = [default_no_refusal(), default_non_empty()]
    if strategy == "rule":
        judge_obj: Judge = RuleJudge(checks=default_checks)
    elif strategy == "llm":
        provider_obj = _resolve_provider(provider)
        judge_obj = LLMJudge(provider=provider_obj)
    else:  # ensemble
        provider_obj = _resolve_provider(provider)
        judge_obj = EnsembleJudge(
            rule_judge=RuleJudge(checks=default_checks),
            fallback=LLMJudge(provider=provider_obj),
        )

    return judge_obj.score(
        rubric=rubric_obj, item=item_obj, target_response=response
    )


def preflight(
    *,
    target: LLMProvider | ProviderSpec | str | dict,
    judge: LLMProvider | ProviderSpec | str | dict,
    profile: ExecutionProfile | str = ExecutionProfile.GUARDED,
) -> PreflightReport:
    """Sanity-check the target / judge environment before calibrating.

    Capability-only check by default: examines provider tiers and known
    placeholder/experimental flags, surfaces self-agreement-bias warnings
    when target and judge are the same vendor.

    If ``mini-omega-lock`` is installed, this entrypoint will also be the
    plug point for empirical probes (judge consistency, endpoint reliability,
    latency) once those are wired through. Standalone callers receive the
    capability-only report.
    """
    profile_obj = (
        profile
        if isinstance(profile, ExecutionProfile)
        else ExecutionProfile(profile)
    )
    target_provider = _resolve_provider(target)
    judge_provider = _resolve_provider(judge)
    target_caps = provider_capabilities(target_provider)
    judge_caps = provider_capabilities(judge_provider)

    warnings: list[str] = []
    blocker_reasons: list[str] = []

    if target_provider.name == judge_provider.name:
        warnings.append(
            f"target and judge share vendor '{target_provider.name}' — "
            "self-agreement bias possible. Cross-vendor judge recommended."
        )
    if getattr(target_caps, "is_placeholder", False):
        blocker_reasons.append(
            f"target provider '{target_provider.name}' is a placeholder; "
            "real provider needed before ship."
        )
    if getattr(judge_caps, "is_placeholder", False):
        blocker_reasons.append(
            f"judge provider '{judge_provider.name}' is a placeholder; "
            "real provider needed before ship."
        )
    if getattr(target_caps, "is_experimental", False):
        warnings.append(
            f"target provider '{target_provider.name}' is experimental — "
            "interface may change."
        )

    if blocker_reasons:
        status = (
            PreflightStatus.ABORT
            if profile_obj == ExecutionProfile.GUARDED
            else PreflightStatus.ADAPT
        )
    elif warnings:
        status = PreflightStatus.PROCEED
    else:
        status = PreflightStatus.PROCEED

    return PreflightReport(
        analytical_findings=[],
        judge_quality=None,
        endpoint=None,
        performance=None,
        status=status,
        blocker_reasons=blocker_reasons,
        warnings=warnings,
    )


def classify_traps(
    *,
    rubric: JudgeRubric | Path | str | dict,
    variants: PromptVariants | Path | str | dict,
    target: LLMProvider | ProviderSpec | str | dict,
    judge: LLMProvider | ProviderSpec | str | dict,
    dataset: Dataset | Path | str,
    test: Dataset | Path | str | None = None,
) -> list[AnalyticalFinding]:
    """Classify the calibration config against known trap patterns.

    Deterministic, zero-LLM-cost. Use case: pre-flight check before
    calibrating — catch self-agreement bias, small-sample power loss,
    variant homogeneity, rubric concentration, etc.

    Requires ``mini-antemortem-cli`` to be installed (the analytical
    sub-tool that ships the trap-pattern classifier).
    """
    try:
        from mini_antemortem_cli import analytical_preflight
    except ImportError as exc:
        raise ImportError(
            "omegaprompt.classify_traps requires mini-antemortem-cli. "
            "Install with `pip install mini-antemortem-cli`."
        ) from exc

    rubric_obj = _resolve_rubric(rubric)
    variants_obj = _resolve_variants(variants)
    target_provider = _resolve_provider(target)
    judge_provider = _resolve_provider(judge)
    train_dataset = _resolve_dataset(dataset)
    test_dataset = _resolve_dataset(test) if test is not None else None

    return analytical_preflight(
        target_provider=target_provider.name,
        target_model=target_provider.model,
        judge_provider=judge_provider.name,
        judge_model=judge_provider.model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        rubric=rubric_obj,
        variants=variants_obj,
    )


# ----------------------------------------------------------------------
# Internal helpers — extracted from commands/calibrate.py to keep the
# pure pipeline orchestration free of typer/CLI concerns
# ----------------------------------------------------------------------


def _sensitivity_rows(stress: object) -> list[dict]:
    if not isinstance(stress, list):
        return []
    ranked = sorted(
        (e for e in stress if isinstance(e, dict)),
        key=lambda e: float(e.get("normalized_stress", 0.0) or 0.0),
        reverse=True,
    )
    return [
        {
            "axis": e.get("name"),
            "gini_delta": e.get("normalized_stress"),
            "raw_stress": e.get("raw_stress"),
            "rank": i,
        }
        for i, e in enumerate(ranked)
    ]


def _per_item_scores(result: object) -> dict[str, float]:
    metadata = getattr(result, "metadata", None)
    rows = metadata.get("per_item_scores", []) if metadata else []
    return {row["item_id"]: float(row["final_score"]) for row in rows}


def _dedupe_events(events: list) -> list:
    seen: set = set()
    unique: list = []
    for e in events:
        key = (e.capability, e.requested, e.applied, e.reason)
        if key in seen:
            continue
        seen.add(key)
        unique.append(e)
    return unique


def _merge_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    merged = dict(left or {})
    for k, v in (right or {}).items():
        merged[k] = int(merged.get(k, 0) + int(v or 0))
    return merged


__all__ = [
    # Tier 1 types + entrypoints
    "ProviderSpec",
    "CalibrateTuning",
    "ArtifactDiff",
    "calibrate",
    "evaluate",
    "report",
    "diff",
    # Tier 2 types + entrypoints
    "SensitivityTuning",
    "SensitivityResult",
    "measure_sensitivity",
    "grade",
    "preflight",
    "classify_traps",
]
