"""``omegaprompt calibrate`` — end-to-end prompt calibration.

Reviewer P0: pre-fix this CLI re-implemented the calibration pipeline,
so the CLI / Python runtime / MCP all had subtly different gate
policies. ``runtime.calibrate()`` is now the canonical pipeline; this
module is the Typer wrapper that parses CLI args, builds
``CalibrateTuning`` + ``ProviderSpec``, and delegates.

The behaviour is the same as ``runtime.calibrate(...)``: a
``CalibrationArtifact`` is written to ``--output`` and a one-line
status summary prints to stdout. New flags exposed through CLI:

- ``--validation-mode {auto,paired,disjoint}`` (already on
  CalibrateTuning since v1.5)
- ``--adaptation-plan PATH`` (loads a saved AdaptationPlan JSON and
  threads it through runtime.calibrate)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import typer

from omegaprompt.domain.profiles import ExecutionProfile
from omegaprompt.preflight import AdaptationPlan
from omegaprompt.providers.factory import supported_providers
from omegaprompt.runtime import (
    CalibrateTuning,
    ProviderSpec,
    calibrate as runtime_calibrate,
)


_ENV_KEYS_FOR_PROVIDER: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}


def _require_env_for(provider_key: str, base_url: str | None = None) -> str | None:
    expected = _ENV_KEYS_FOR_PROVIDER.get(provider_key)
    if not expected:
        return None
    if base_url and "localhost" in base_url:
        return None
    if not any(os.getenv(env) for env in expected):
        return (
            f"{' or '.join(expected)} is not set. Export one before running "
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
    validation_mode: str = typer.Option(  # noqa: B008
        "auto",
        "--validation-mode",
        help=(
            "How walk-forward interprets train/test slices: 'auto' (default — "
            "compute KC-4 only when slices share >=3 ids), 'paired' (assert "
            "shared ids; raise if overlap < 3), 'disjoint' (assert no shared "
            "ids; KC-4 not computed)."
        ),
        case_sensitive=False,
    ),
    adaptation_plan_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--adaptation-plan",
        help=(
            "Optional path to a serialized AdaptationPlan JSON. When supplied, "
            "the plan's overrides tighten min_kc4/max_gap/unlock_k before the "
            "search runs and the plan's manual-review escalation flows into "
            "the artifact's status / ship_recommendation."
        ),
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Calibrate a prompt configuration against a dataset.

    Thin wrapper over ``runtime.calibrate()``. All gate policy
    decisions (KC-4 thresholds, walk-forward semantics, adaptation
    plan honouring, ship recommendation, status escalation) live in
    the runtime so CLI / Python / MCP behave identically.
    """
    selected_profile = ExecutionProfile(profile)

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

    for provider_key, base_url in {tp: target_base_url, jp: judge_base_url}.items():
        err = _require_env_for(provider_key, base_url=base_url)
        if err:
            typer.secho(err, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2)

    vmode = validation_mode.lower().strip()
    if vmode not in {"auto", "paired", "disjoint"}:
        typer.secho(
            f"Unknown --validation-mode {validation_mode!r}. "
            "Use one of: auto, paired, disjoint.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    space = None
    if space_path is not None:
        from omegaprompt.domain.params import MetaAxisSpace
        space = MetaAxisSpace.model_validate_json(
            space_path.read_text(encoding="utf-8")
        )

    plan: AdaptationPlan | None = None
    if adaptation_plan_path is not None:
        plan = AdaptationPlan.model_validate_json(
            adaptation_plan_path.read_text(encoding="utf-8")
        )

    tuning = CalibrateTuning(
        method=method.lower().strip(),
        unlock_k=unlock_k,
        space=space,
        max_gap=max_gap,
        min_kc4=min_kc4,
        profile=selected_profile,
        validation_mode=vmode,  # type: ignore[arg-type]
    )

    typer.secho(
        f"Target: {tp}/{target_model or 'default'}   "
        f"Judge: {jp}/{judge_model or 'default'}",
        fg=typer.colors.BRIGHT_BLACK,
    )
    typer.secho(
        f"Profile: {selected_profile.value}   method: {tuning.method}   "
        f"unlock_k={tuning.unlock_k}   validation_mode={vmode}",
        fg=typer.colors.BRIGHT_BLACK,
    )

    artifact = runtime_calibrate(
        train=dataset_path,
        rubric=rubric_path,
        variants=variants_path,
        target=ProviderSpec(name=tp, model=target_model, base_url=target_base_url),
        judge=ProviderSpec(name=jp, model=judge_model, base_url=judge_base_url),
        test=test_path,
        output=output_path,
        tuning=tuning,
        adaptation_plan=plan,
    )

    color = (
        typer.colors.GREEN
        if artifact.status == "OK"
        else typer.colors.RED
    )
    typer.secho(
        f"\n{artifact.status}: calibrated_fitness={artifact.calibrated_fitness:.4f} "
        f"neutral_fitness={artifact.neutral_fitness:.4f} "
        f"uplift={artifact.uplift_percent:+.1f}%",
        fg=color,
    )
    typer.secho(
        f"Ship recommendation: {artifact.ship_recommendation.value}   "
        f"Hard-gate pass rate: {artifact.hard_gate_pass_rate:.1%}   "
        f"Stayed within guarded boundaries: {artifact.stayed_within_guarded_boundaries}",
        fg=typer.colors.BRIGHT_BLACK,
    )
    typer.secho(f"Artifact written to: {output_path}", fg=typer.colors.BRIGHT_BLACK)
    if artifact.status != "OK":
        raise typer.Exit(code=1)
