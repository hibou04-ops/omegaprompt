"""FastMCP server wrapping the eight runtime entrypoints.

Each tool accepts JSON-friendly input (str paths, dicts, primitives) and
returns the runtime result serialized as a dict / list / str. Internal
coercion via ``omegaprompt.runtime``'s ``_resolve_*`` helpers means agents
can pass either filesystem paths or inline JSON for datasets, rubrics,
variants, and provider configurations.

The agent-facing tool descriptions are intentionally short and action-
oriented; the long-form rationale lives in the runtime function docstrings.
"""

from __future__ import annotations

from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from omegaprompt import runtime

mcp_app = FastMCP(
    name="omegaprompt",
    instructions=(
        "Provider-neutral prompt calibration and validation. Use these tools "
        "before shipping a prompt to catch overfitting (walk-forward gate), "
        "self-agreement bias (cross-vendor judging), and silent provider "
        "degradation (capability events). The artifact each tool produces "
        "is the canonical record — pass it back into other tools rather "
        "than re-running."
    ),
)


# ---------------------------------------------------------------------------
# Tier 1: most-used entrypoints
# ---------------------------------------------------------------------------


@mcp_app.tool()
def calibrate(
    train: str,
    rubric: str | dict,
    variants: str | dict,
    target: str | dict,
    judge: str | dict | None = None,
    test: str | None = None,
    output: str | None = None,
    tuning: dict | None = None,
    adaptation_plan: dict | None = None,
) -> dict:
    """Run the full calibration pipeline and return the resulting artifact.

    Use when validating a prompt configuration before ship. Expensive
    (dozens to hundreds of LLM calls). Use ``classify_traps`` and
    ``preflight`` first to catch cheap-to-fix issues.

    Args:
        train: Path to a training-dataset JSONL file.
        rubric: Path to a JudgeRubric JSON, or an inline rubric dict.
        variants: Path to a PromptVariants JSON, or an inline variants dict.
        target: Provider name (``"anthropic"``, ``"openai"``, ``"local"``)
            or a ``ProviderSpec`` dict ``{"name": ..., "model": ..., "base_url": ...}``.
        judge: Provider for the judge calls. If omitted, ``target`` is reused
            (enables self-agreement bias — pass a different vendor for a
            stronger signal).
        test: Optional path to a held-out test JSONL. Strongly recommended;
            without it the walk-forward gate cannot run.
        output: If set, also writes the artifact JSON to this path.
        tuning: Optional ``CalibrateTuning`` dict (method, unlock_k, space,
            max_gap, min_kc4, profile, validation_mode). Defaults from the
            execution profile.
        adaptation_plan: Optional serialized ``AdaptationPlan`` dict from
            a prior preflight run. The plan's overrides tighten min_kc4,
            max_gap, and unlock_k before the search runs; manual-review
            escalation flows into the artifact's status / ship_recommendation.
            Reaches CLI/Python parity (Reviewer P0 #4).

    Returns:
        CalibrationArtifact serialized as a dict. Inspect ``status`` and
        ``ship_recommendation`` to decide whether to ship.
    """
    plan_obj = None
    if adaptation_plan is not None:
        from omegaprompt.preflight import AdaptationPlan
        plan_obj = AdaptationPlan(**adaptation_plan)

    artifact = runtime.calibrate(
        train=train,
        rubric=rubric,
        variants=variants,
        target=target,
        judge=judge,
        test=test,
        output=output,
        tuning=runtime.CalibrateTuning(**tuning) if tuning else None,
        adaptation_plan=plan_obj,
    )
    return artifact.model_dump(mode="json")


@mcp_app.tool()
def evaluate(
    dataset: str,
    rubric: str | dict,
    variants: str | dict,
    params: str | dict,
    target: str | dict,
    judge: str | dict | None = None,
    profile: Literal["guarded", "expedition"] = "guarded",
) -> dict:
    """Score a fixed prompt configuration on a dataset (no search).

    Use for regression checks on a previously calibrated config, or to
    cheaply score a candidate before deciding whether full calibration
    is worth the cost.

    Args:
        dataset: Path to a JSONL dataset.
        rubric: Path or inline dict for the JudgeRubric.
        variants: Path or inline dict for the PromptVariants.
        params: Either an inline ``ResolvedPromptParams`` dict, or the path
            to a CalibrationArtifact JSON (in which case its
            ``calibrated_params`` is extracted automatically).
        target: Target provider (string or ProviderSpec dict).
        judge: Judge provider; defaults to the target.
        profile: Execution profile, ``"guarded"`` (default) or ``"expedition"``.

    Returns:
        EvalResult serialized as a dict (fitness, hard_gate_pass_rate,
        per-item scores, capability events).
    """
    if isinstance(params, str):
        from omegaprompt.core.artifact import load_artifact

        params_resolved = load_artifact(params)
    else:
        params_resolved = params

    result = runtime.evaluate(
        dataset=dataset,
        rubric=rubric,
        variants=variants,
        params=params_resolved,
        target=target,
        judge=judge,
        profile=profile,
    )
    return result.model_dump(mode="json")


@mcp_app.tool()
def report(artifact: str) -> str:
    """Render a CalibrationArtifact as Markdown for human review.

    Args:
        artifact: Path to a CalibrationArtifact JSON.

    Returns:
        Markdown summary string. Suitable for posting to PR descriptions,
        CI step output, or chat messages.
    """
    return runtime.report(artifact)


@mcp_app.tool()
def diff(
    baseline: str,
    candidate: str,
    format: Literal["json", "markdown"] = "json",
) -> dict | str:
    """Compare two CalibrationArtifacts and report regressions.

    Args:
        baseline: Path to the OLD artifact JSON (canonical baseline).
        candidate: Path to the NEW artifact JSON.
        format: ``"json"`` (default) returns a structured diff dict;
            ``"markdown"`` returns a human-readable string.

    Returns:
        ArtifactDiff dict or markdown string. ``regressed=true`` (in the
        dict form) signals a breaking change; ``regression_reasons`` lists
        which metrics dropped.
    """
    result = runtime.diff(baseline, candidate, format=format)
    if isinstance(result, str):
        return result
    return result.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Tier 2: less-frequent but distinct entrypoints
# ---------------------------------------------------------------------------


@mcp_app.tool()
def measure_sensitivity(
    dataset: str,
    rubric: str | dict,
    variants: str | dict,
    target: str | dict,
    judge: str | dict | None = None,
    tuning: dict | None = None,
) -> dict:
    """Cheap axis-ranking probe: which meta-axes carry signal on this task?

    Use as a cheap pre-flight before deciding whether full ``calibrate`` is
    worth the cost. If no axis has meaningful normalized_stress, calibrate
    will not produce uplift; if 2-3 axes dominate, calibrate is on solid
    ground.

    Args:
        dataset: Path to a (small) probe-dataset JSONL.
        rubric: Path or inline dict for the JudgeRubric.
        variants: Path or inline dict for the PromptVariants.
        target: Target provider.
        judge: Judge provider; defaults to target.
        tuning: Optional ``SensitivityTuning`` dict (space, profile).

    Returns:
        SensitivityResult dict with per-axis ranked stress scores.
    """
    result = runtime.measure_sensitivity(
        dataset=dataset,
        rubric=rubric,
        variants=variants,
        target=target,
        judge=judge,
        tuning=runtime.SensitivityTuning(**tuning) if tuning else None,
    )
    return result.model_dump(mode="json")


@mcp_app.tool()
def grade(
    rubric: str | dict,
    item: dict,
    response: str,
    provider: str | dict,
    strategy: Literal["rule", "llm", "ensemble"] = "ensemble",
) -> dict:
    """Score one response against a rubric.

    Use for self-grading an agent output before returning it, or for
    spot-checking a candidate without running a full evaluation.

    Args:
        rubric: Path or inline dict for the JudgeRubric.
        item: Inline DatasetItem dict ``{"id": ..., "input": ...,
            "reference": ...}``.
        response: The candidate response string to score.
        provider: Provider used for LLM judging (ignored if strategy="rule").
        strategy: ``"rule"`` (deterministic, zero-API), ``"llm"`` (single
            LLM call), ``"ensemble"`` (default — rule first, escalate to
            LLM only if hard gates pass).

    Returns:
        JudgeResult dict (per-dimension scores, hard-gate flags, rationale).
    """
    result = runtime.grade(
        rubric=rubric,
        item=item,
        response=response,
        provider=provider,
        strategy=strategy,
    )
    return result.model_dump(mode="json")


@mcp_app.tool()
def preflight(
    target: str | dict,
    judge: str | dict,
    profile: Literal["guarded", "expedition"] = "guarded",
) -> dict:
    """Sanity-check the target/judge environment before calibrating.

    Capability-only check: surfaces self-agreement-bias warnings when target
    and judge share vendor; blocks on placeholder providers under guarded
    profile. If ``mini-omega-lock`` is installed, this entrypoint is the
    plug point for empirical probes (judge consistency, endpoint reliability,
    latency).

    Args:
        target: Target provider (string or ProviderSpec dict).
        judge: Judge provider.
        profile: ``"guarded"`` (default) or ``"expedition"``.

    Returns:
        PreflightReport dict. ``status="abort"`` means do not proceed.
    """
    result = runtime.preflight(target=target, judge=judge, profile=profile)
    return result.model_dump(mode="json")


@mcp_app.tool()
def classify_traps(
    rubric: str | dict,
    variants: str | dict,
    target: str | dict,
    judge: str | dict,
    dataset: str,
    test: str | None = None,
) -> list[dict]:
    """Classify the calibration config against known trap patterns.

    Deterministic, zero-LLM-cost. Catches self-agreement bias, small-sample
    KC-4 power loss, variant homogeneity, rubric concentration, judge
    budget shortfalls, empty references, and missing held-out slices.
    Requires ``mini-antemortem-cli`` to be installed.

    Args:
        rubric: Path or inline dict for the JudgeRubric.
        variants: Path or inline dict for the PromptVariants.
        target: Target provider.
        judge: Judge provider.
        dataset: Path to the train dataset JSONL.
        test: Optional path to the held-out test JSONL.

    Returns:
        List of AnalyticalFinding dicts. Each has ``trap_id``, ``label``
        (REAL / GHOST / NEW / UNRESOLVED), ``severity``, ``note``, and
        ``remediation``.
    """
    findings = runtime.classify_traps(
        rubric=rubric,
        variants=variants,
        target=target,
        judge=judge,
        dataset=dataset,
        test=test,
    )
    return [f.model_dump(mode="json") for f in findings]
