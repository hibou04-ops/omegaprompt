"""``omegacal calibrate`` - end-to-end prompt calibration."""

from __future__ import annotations

import os
from pathlib import Path

import typer

from omegaprompt.core import (
    assess_run_risk,
    evaluate_walk_forward,
    policy_for,
    relaxed_safeguards_for,
    save_artifact,
)
from omegaprompt.domain.dataset import Dataset
from omegaprompt.domain.judge import JudgeRubric
from omegaprompt.domain.params import MetaAxisSpace, PromptVariants
from omegaprompt.domain.profiles import ExecutionProfile
from omegaprompt.domain.result import CalibrationArtifact
from omegaprompt.judges.llm_judge import LLMJudge
from omegaprompt.providers import (
    ProviderError,
    provider_capabilities,
    quality_per_cost,
    quality_per_latency,
)
from omegaprompt.providers.factory import make_provider, supported_providers
from omegaprompt.targets.prompt_target import PromptTarget


_ENV_KEY_FOR_PROVIDER: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _require_env_for(provider_key: str, base_url: str | None = None) -> str | None:
    expected = _ENV_KEY_FOR_PROVIDER.get(provider_key)
    if expected is None:
        return None
    if base_url and "localhost" in base_url:
        return None
    if not os.getenv(expected):
        return (
            f"{expected} is not set. Export it before running "
            f"`omegacal calibrate` with provider={provider_key!r}."
        )
    return None


def calibrate(
    dataset_path: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path to the training dataset (.jsonl).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    rubric_path: Path = typer.Option(  # noqa: B008
        ...,
        "--rubric",
        "-r",
        help="Path to the JudgeRubric JSON.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    variants_path: Path = typer.Option(  # noqa: B008
        ...,
        "--variants",
        "-v",
        help="Path to the PromptVariants JSON (system_prompts + few_shot_examples).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    test_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--test",
        "-t",
        help="Path to the walk-forward test dataset (.jsonl). Strongly recommended.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_path: Path = typer.Option(  # noqa: B008
        Path("calibration_outcome.json"),
        "--output",
        "-o",
        help="Where to write the CalibrationArtifact JSON.",
        file_okay=True,
        dir_okay=False,
    ),
    profile: ExecutionProfile = typer.Option(  # noqa: B008
        ExecutionProfile.GUARDED,
        "--profile",
        help="Execution profile: guarded (default) or expedition.",
        case_sensitive=False,
    ),
    target_provider: str = typer.Option(  # noqa: B008
        "anthropic",
        "--target-provider",
        help="Provider for the target calls. One of: " + ", ".join(supported_providers()) + ".",
        case_sensitive=False,
    ),
    target_model: str | None = typer.Option(  # noqa: B008
        None,
        "--target-model",
        help="Target model string. If omitted, uses the provider default.",
    ),
    target_base_url: str | None = typer.Option(  # noqa: B008
        None,
        "--target-base-url",
        help="Custom endpoint for OpenAI-compatible or local target providers.",
    ),
    judge_provider: str = typer.Option(  # noqa: B008
        "anthropic",
        "--judge-provider",
        help=(
            "Provider for the LLM-judge calls. Can differ from --target-provider. "
            "One of: " + ", ".join(supported_providers()) + "."
        ),
        case_sensitive=False,
    ),
    judge_model: str | None = typer.Option(  # noqa: B008
        None,
        "--judge-model",
        help="Judge model string. If omitted, uses the provider default.",
    ),
    judge_base_url: str | None = typer.Option(  # noqa: B008
        None,
        "--judge-base-url",
        help="Custom endpoint for OpenAI-compatible or local judge providers.",
    ),
    method: str = typer.Option(  # noqa: B008
        "p1",
        "--method",
        "-m",
        help="Calibration method: 'p1' (stress + top-K unlock + grid + walk-forward replay).",
        case_sensitive=False,
    ),
    unlock_k: int = typer.Option(  # noqa: B008
        3,
        "--unlock-k",
        min=1,
        help="How many top-stress meta-axes to unlock for grid search.",
    ),
    space_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--space",
        help="Optional MetaAxisSpace JSON to override the default meta-axis bounds.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    max_gap: float | None = typer.Option(  # noqa: B008
        None,
        "--max-gap",
        help="Walk-forward: maximum allowed |train - test| / |train| before FAIL_KC4_GATE.",
    ),
    min_kc4: float | None = typer.Option(  # noqa: B008
        None,
        "--min-kc4",
        help="Walk-forward: minimum Pearson correlation when computable.",
    ),
) -> None:
    """Calibrate a prompt configuration against a dataset."""

    selected_profile = ExecutionProfile(profile)
    policy = policy_for(selected_profile)
    resolved_max_gap = max_gap if max_gap is not None else policy.default_max_gap
    resolved_min_kc4 = min_kc4 if min_kc4 is not None else policy.default_min_kc4

    tp = target_provider.lower().strip()
    jp = judge_provider.lower().strip()
    if tp not in supported_providers():
        typer.secho(
            f"Unknown --target-provider {target_provider!r}. Supported: "
            + ", ".join(supported_providers()),
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    if jp not in supported_providers():
        typer.secho(
            f"Unknown --judge-provider {judge_provider!r}. Supported: "
            + ", ".join(supported_providers()),
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    provider_envs = {
        tp: target_base_url,
        jp: judge_base_url,
    }
    for provider_key, base_url in provider_envs.items():
        err = _require_env_for(provider_key, base_url=base_url)
        if err:
            typer.secho(err, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2)

    try:
        from omega_lock import P1Config, run_p1  # type: ignore
    except ImportError as exc:
        typer.secho(
            f"The 'omega-lock' package is required for `calibrate`. "
            f"Install with `pip install omega-lock`. ({exc})",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2) from exc

    train_ds = Dataset.from_jsonl(dataset_path)
    test_ds = Dataset.from_jsonl(test_path) if test_path is not None else None
    rubric = JudgeRubric.from_json(rubric_path)
    variants = PromptVariants.model_validate_json(variants_path.read_text(encoding="utf-8"))
    space = (
        MetaAxisSpace.model_validate_json(space_path.read_text(encoding="utf-8"))
        if space_path is not None
        else None
    )

    try:
        target_provider_obj = make_provider(tp, model=target_model, base_url=target_base_url)
        judge_provider_obj = make_provider(jp, model=judge_model, base_url=judge_base_url)
    except ProviderError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    judge = LLMJudge(provider=judge_provider_obj, execution_profile=selected_profile)
    train_target = PromptTarget(
        target_provider=target_provider_obj,
        judge=judge,
        dataset=train_ds,
        rubric=rubric,
        variants=variants,
        space=space,
        execution_profile=selected_profile,
    )
    test_target = (
        PromptTarget(
            target_provider=target_provider_obj,
            judge=judge,
            dataset=test_ds,
            rubric=rubric,
            variants=variants,
            space=space,
            execution_profile=selected_profile,
        )
        if test_ds is not None
        else None
    )

    typer.secho(
        f"Target: {target_provider_obj.name}/{target_provider_obj.model}   "
        f"Judge: {judge_provider_obj.name}/{judge_provider_obj.model}",
        fg=typer.colors.BRIGHT_BLACK,
    )
    typer.secho(
        f"Profile: {selected_profile.value}   method: {method}   unlock_k={unlock_k}",
        fg=typer.colors.BRIGHT_BLACK,
    )

    neutral_result = train_target.evaluate(train_target.neutral_params())

    config = P1Config(unlock_k=unlock_k)
    result = run_p1(train_target=train_target, test_target=test_target, config=config)

    grid_best = getattr(result, "grid_best", None) or {}
    best_unlocked: dict = grid_best.get("unlocked", {}) if isinstance(grid_best, dict) else {}
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
                f"train={best_train_eval.fitness:.3f} test={test_eval.fitness:.3f} "
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
    target_caps = provider_capabilities(target_provider_obj)
    judge_caps = provider_capabilities(judge_provider_obj)
    boundary_warnings, within_guarded, ship_recommendation = assess_run_risk(
        profile=selected_profile,
        target_capabilities=target_caps,
        judge_capabilities=judge_caps,
        degraded_capabilities=degraded_capabilities,
        has_walk_forward=test_target is not None,
        walk_forward_passed=None if walk_forward is None else walk_forward.passed,
    )

    calibrated_fitness = best_train_eval.fitness
    neutral_fitness = neutral_result.fitness
    uplift_absolute = calibrated_fitness - neutral_fitness
    uplift_percent = (uplift_absolute / neutral_fitness * 100.0) if neutral_fitness else 0.0

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
        method=method,
        unlock_k=unlock_k,
        selected_profile=selected_profile,
        neutral_baseline_params=neutral_result.resolved_params,
        calibrated_params=best_train_eval.resolved_params,
        neutral_fitness=neutral_fitness,
        calibrated_fitness=calibrated_fitness,
        uplift_absolute=uplift_absolute,
        uplift_percent=uplift_percent,
        quality_per_cost_neutral=quality_per_cost(neutral_fitness, neutral_result.estimated_cost_units),
        quality_per_cost_best=quality_per_cost(calibrated_fitness, best_train_eval.estimated_cost_units),
        quality_per_latency_neutral=quality_per_latency(neutral_fitness, neutral_result.latency_ms),
        quality_per_latency_best=quality_per_latency(calibrated_fitness, best_train_eval.latency_ms),
        boundary_warnings=boundary_warnings,
        degraded_capabilities=degraded_capabilities,
        ship_recommendation=ship_recommendation,
        stayed_within_guarded_boundaries=within_guarded and best_train_eval.within_guarded_boundaries,
        additional_uplift_from_boundary_crossing=additional_uplift,
        relaxed_safeguards=relaxed_safeguards_for(selected_profile),
        guarded_boundary_crossed=not (within_guarded and best_train_eval.within_guarded_boundaries),
        best_params=best_train_eval.resolved_params,
        best_fitness=calibrated_fitness,
        walk_forward=walk_forward,
        hard_gate_pass_rate=best_train_eval.hard_gate_pass_rate,
        sensitivity_ranking=sensitivity_rows,
        n_candidates_evaluated=train_target.unique_param_count(),
        total_api_calls=train_target.total_api_calls + (test_target.total_api_calls if test_target else 0),
        usage_summary=_merge_usage(
            train_target.last_usage,
            test_target.last_usage if test_target is not None else {},
        ),
        latency_summary_ms={
            "neutral_train": neutral_result.latency_ms,
            "calibrated_train": best_train_eval.latency_ms,
            "calibrated_test": test_eval.latency_ms if test_eval is not None else 0.0,
        },
        target_provider=target_provider_obj.name,
        target_model=target_provider_obj.model,
        judge_provider=judge_provider_obj.name,
        judge_model=judge_provider_obj.model,
        target_capabilities=target_caps,
        judge_capabilities=judge_caps,
        status=status,
        rationale=rationale,
    )

    save_artifact(artifact, output_path)
    colour = typer.colors.GREEN if status == "OK" else typer.colors.YELLOW
    typer.secho(
        f"Calibration complete [{status}]. calibrated_fitness={calibrated_fitness:.4f}",
        fg=colour,
    )
    typer.secho(
        f"neutral={neutral_fitness:.4f} uplift={uplift_absolute:+.4f} ({uplift_percent:+.2f}%)",
        fg=colour,
    )
    typer.secho(f"Artifact: {output_path}", fg=typer.colors.GREEN)


def _sensitivity_rows(stress: object) -> list[dict]:
    sensitivity_rows: list[dict] = []
    if not isinstance(stress, list):
        return sensitivity_rows
    ranked = sorted(
        (entry for entry in stress if isinstance(entry, dict)),
        key=lambda entry: float(entry.get("normalized_stress", 0.0) or 0.0),
        reverse=True,
    )
    for rank, entry in enumerate(ranked):
        sensitivity_rows.append(
            {
                "axis": entry.get("name"),
                "gini_delta": entry.get("normalized_stress"),
                "raw_stress": entry.get("raw_stress"),
                "rank": rank,
            }
        )
    return sensitivity_rows


def _per_item_scores(result) -> dict[str, float]:
    rows = result.metadata.get("per_item_scores", []) if getattr(result, "metadata", None) else []
    return {row["item_id"]: float(row["final_score"]) for row in rows}


def _dedupe_events(events):
    seen = set()
    unique = []
    for event in events:
        key = (event.capability, event.requested, event.applied, event.reason)
        if key in seen:
            continue
        seen.add(key)
        unique.append(event)
    return unique


def _merge_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    merged = dict(left or {})
    for key, value in (right or {}).items():
        merged[key] = int(merged.get(key, 0) + int(value or 0))
    return merged
