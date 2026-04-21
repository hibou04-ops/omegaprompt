"""``omegaprompt calibrate`` - end-to-end prompt calibration.

Model-agnostic: target and judge providers are selected independently.
The calibration pipeline (stress, grid, walk-forward, KC-4) is
vendor-neutral; only the two API-call boundaries touch a vendor SDK.

A strong pattern for cross-vendor validation: use different providers for
target and judge. Example: target = your production prompt on gpt-4o,
judge = claude-opus as a higher-capability grader.
"""

from __future__ import annotations

import os
from pathlib import Path

import typer

from omegaprompt.core.artifact import save_artifact
from omegaprompt.core.walkforward import evaluate_walk_forward
from omegaprompt.domain.dataset import Dataset
from omegaprompt.domain.judge import JudgeRubric
from omegaprompt.domain.params import MetaAxisSpace, PromptVariants
from omegaprompt.domain.result import CalibrationArtifact
from omegaprompt.judges.llm_judge import LLMJudge
from omegaprompt.providers import (
    DEFAULT_MODELS,
    ProviderError,
    make_provider,
    supported_providers,
)
from omegaprompt.targets.prompt_target import PromptTarget


_ENV_KEY_FOR_PROVIDER: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _require_env_for(provider_key: str) -> str | None:
    expected = _ENV_KEY_FOR_PROVIDER.get(provider_key)
    if expected is None:
        return None
    if not os.getenv(expected):
        return (
            f"{expected} is not set. Export it before running "
            f"`omegaprompt calibrate` with provider={provider_key!r}."
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
        help="Path to the walk-forward test dataset (.jsonl). Recommended.",
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
    target_provider: str = typer.Option(  # noqa: B008
        "anthropic",
        "--target-provider",
        help="Provider for the target calls. One of: " + ", ".join(supported_providers()) + ".",
        case_sensitive=False,
    ),
    target_model: str | None = typer.Option(  # noqa: B008
        None,
        "--target-model",
        help=(
            "Target model string. If omitted, uses the target provider's default "
            f"(anthropic={DEFAULT_MODELS['anthropic']}, openai={DEFAULT_MODELS['openai']})."
        ),
    ),
    target_base_url: str | None = typer.Option(  # noqa: B008
        None,
        "--target-base-url",
        help="Custom endpoint for OpenAI-compatible target providers (Azure, Groq, Ollama, ...).",
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
        help="Judge model string. If omitted, uses the judge provider's default.",
    ),
    judge_base_url: str | None = typer.Option(  # noqa: B008
        None,
        "--judge-base-url",
        help="Custom endpoint for OpenAI-compatible judge providers.",
    ),
    method: str = typer.Option(  # noqa: B008
        "p1",
        "--method",
        "-m",
        help="Calibration method: 'p1' (omega-lock P1: stress + grid + KC-4).",
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
    max_gap: float = typer.Option(  # noqa: B008
        0.25,
        "--max-gap",
        help="Walk-forward: maximum allowed |train - test| / |train| before FAIL_KC4_GATE.",
    ),
    min_kc4: float = typer.Option(  # noqa: B008
        0.5,
        "--min-kc4",
        help="Walk-forward: minimum Pearson correlation when computable.",
    ),
) -> None:
    """Calibrate a prompt configuration against a dataset."""

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

    for provider_key in {tp, jp}:
        err = _require_env_for(provider_key)
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

    typer.secho(f"Loading dataset from {dataset_path} ...", fg=typer.colors.BRIGHT_BLACK)
    train_ds = Dataset.from_jsonl(dataset_path)
    typer.secho(f"  {len(train_ds)} items", fg=typer.colors.BRIGHT_BLACK)

    test_ds: Dataset | None = None
    if test_path is not None:
        typer.secho(f"Loading test set from {test_path} ...", fg=typer.colors.BRIGHT_BLACK)
        test_ds = Dataset.from_jsonl(test_path)
        typer.secho(f"  {len(test_ds)} items", fg=typer.colors.BRIGHT_BLACK)

    rubric = JudgeRubric.from_json(rubric_path)
    variants = PromptVariants.model_validate_json(variants_path.read_text(encoding="utf-8"))
    space: MetaAxisSpace | None = None
    if space_path is not None:
        space = MetaAxisSpace.model_validate_json(space_path.read_text(encoding="utf-8"))

    try:
        target_provider_obj = make_provider(tp, model=target_model, base_url=target_base_url)
        judge_provider_obj = make_provider(jp, model=judge_model, base_url=judge_base_url)
    except ProviderError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    judge = LLMJudge(provider=judge_provider_obj)

    train_target = PromptTarget(
        target_provider=target_provider_obj,
        judge=judge,
        dataset=train_ds,
        rubric=rubric,
        variants=variants,
        space=space,
    )
    test_target: PromptTarget | None = None
    if test_ds is not None:
        test_target = PromptTarget(
            target_provider=target_provider_obj,
            judge=judge,
            dataset=test_ds,
            rubric=rubric,
            variants=variants,
            space=space,
        )

    typer.secho(
        f"Target: {target_provider_obj.name}/{target_provider_obj.model}   "
        f"Judge: {judge_provider_obj.name}/{judge_provider_obj.model}",
        fg=typer.colors.BRIGHT_BLACK,
    )
    typer.secho(
        f"Starting {method} calibration (unlock_k={unlock_k}) ...",
        fg=typer.colors.BRIGHT_BLACK,
    )
    typer.secho("This issues LLM API calls. Budget accordingly.", fg=typer.colors.YELLOW)

    config = P1Config(unlock_k=unlock_k)
    result = run_p1(
        train_target=train_target,
        test_target=test_target,
        config=config,
    )

    grid_best = getattr(result, "grid_best", None) or {}
    best_unlocked: dict = (
        grid_best.get("unlocked", {}) if isinstance(grid_best, dict) else {}
    )
    best_fitness = (
        float(grid_best.get("fitness", 0.0)) if isinstance(grid_best, dict) else 0.0
    )

    # omega_lock.P1Result puts test fitnesses in the top-level walk_forward block
    # (``walk_forward["test_fitnesses"][0]`` corresponds to ``grid_best``), NOT on
    # the ``grid_best`` dict itself.
    wf_block = getattr(result, "walk_forward", None) or {}
    test_fitness: float | None = None
    if test_target is not None and isinstance(wf_block, dict):
        test_fits = wf_block.get("test_fitnesses") or []
        if test_fits:
            test_fitness = float(test_fits[0])

    walk_forward = None
    status = "OK"
    rationale = "passed"

    if test_fitness is not None:
        walk_forward = evaluate_walk_forward(
            best_fitness,
            test_fitness,
            max_gap=max_gap,
            min_kc4=min_kc4,
        )
        if not walk_forward.passed:
            status = "FAIL_KC4_GATE"
            rationale = (
                f"train={best_fitness:.3f} test={test_fitness:.3f} "
                f"gap={walk_forward.generalization_gap:.2%} "
                f"kc4={walk_forward.kc4_correlation}"
            )

    # omega_lock.P1Result exposes per-axis stress under ``stress_results``; each
    # entry is ``{"name", "normalized_stress", "raw_stress", ...}``.
    sensitivity_rows: list[dict] = []
    stress = getattr(result, "stress_results", None)
    if isinstance(stress, list):
        ranked = sorted(
            (e for e in stress if isinstance(e, dict)),
            key=lambda e: float(e.get("normalized_stress", 0.0) or 0.0),
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

    artifact = CalibrationArtifact(
        method=method,
        unlock_k=unlock_k,
        best_params=best_unlocked,
        best_fitness=best_fitness,
        walk_forward=walk_forward,
        hard_gate_pass_rate=train_target._fitness.pass_rate(),
        sensitivity_ranking=sensitivity_rows,
        n_candidates_evaluated=train_target.total_api_calls // max(len(train_ds) * 2, 1),
        total_api_calls=train_target.total_api_calls
        + (test_target.total_api_calls if test_target else 0),
        usage_summary=dict(train_target.last_usage),
        target_provider=target_provider_obj.name,
        target_model=target_provider_obj.model,
        judge_provider=judge_provider_obj.name,
        judge_model=judge_provider_obj.model,
        status=status,
        rationale=rationale,
    )

    save_artifact(artifact, output_path)
    colour = typer.colors.GREEN if status == "OK" else typer.colors.YELLOW
    typer.secho(
        f"Calibration complete [{status}]. best_fitness={best_fitness:.4f}"
        + (f", test_fitness={test_fitness:.4f}" if test_fitness is not None else ""),
        fg=colour,
    )
    if rationale and rationale != "passed":
        typer.secho(f"  {rationale}", fg=colour)
    typer.secho(f"Artifact: {output_path}", fg=typer.colors.GREEN)
