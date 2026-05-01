"""Profile policy and beginner-facing structural risk tests."""

from __future__ import annotations

from omegaprompt.core.profiles import policy_for
from omegaprompt.core.risk import assess_run_risk
from omegaprompt.domain.profiles import ExecutionProfile, ShipRecommendation
from omegaprompt.domain.result import CalibrationArtifact
from omegaprompt.providers.base import CapabilityEvent, CapabilityTier, ProviderCapabilities
from omegaprompt.reporting.markdown import render_markdown


def _caps(
    *,
    provider: str,
    tier: CapabilityTier,
    ship_grade_judge: bool,
    experimental: bool = False,
    placeholder: bool = False,
) -> ProviderCapabilities:
    return ProviderCapabilities(
        provider=provider,
        tier=tier,
        supports_strict_schema=ship_grade_judge,
        supports_json_object=True,
        supports_reasoning_profiles=True,
        supports_usage_accounting=True,
        supports_llm_judge=ship_grade_judge,
        ship_grade_judge=ship_grade_judge,
        experimental=experimental,
        placeholder=placeholder,
    )


def test_guarded_profile_blocks_weak_judge_and_missing_walk_forward():
    warnings, within_guarded, recommendation = assess_run_risk(
        profile=ExecutionProfile.GUARDED,
        target_capabilities=_caps(provider="ollama", tier=CapabilityTier.LOCAL, ship_grade_judge=False, experimental=True),
        judge_capabilities=_caps(provider="ollama", tier=CapabilityTier.LOCAL, ship_grade_judge=False, experimental=True),
        degraded_capabilities=[],
        has_walk_forward=False,
        walk_forward_passed=None,
    )
    assert within_guarded is False
    assert recommendation == ShipRecommendation.BLOCK
    categories = {warning.category.value for warning in warnings}
    assert "validation strength" in categories
    assert "deployment readiness" in categories


def test_expedition_profile_allows_boundary_crossing_but_marks_experiment():
    warnings, within_guarded, recommendation = assess_run_risk(
        profile=ExecutionProfile.EXPEDITION,
        target_capabilities=_caps(provider="ollama", tier=CapabilityTier.LOCAL, ship_grade_judge=False, experimental=True),
        judge_capabilities=_caps(provider="anthropic", tier=CapabilityTier.CLOUD, ship_grade_judge=True),
        degraded_capabilities=[
            CapabilityEvent(
                capability="structured_output",
                requested="strict_schema",
                applied="json_object_parse",
                reason="local fallback",
                user_visible_note="Validation strength dropped to JSON validation.",
            )
        ],
        has_walk_forward=True,
        walk_forward_passed=True,
    )
    assert recommendation == ShipRecommendation.EXPERIMENT
    assert any(warning.category.value == "safety boundary" for warning in warnings)
    assert within_guarded is False


def test_policy_for_profiles_changes_thresholds():
    guarded = policy_for(ExecutionProfile.GUARDED)
    expedition = policy_for(ExecutionProfile.EXPEDITION)
    assert guarded.allow_schema_degradation is False
    assert expedition.allow_schema_degradation is True
    assert expedition.default_max_gap > guarded.default_max_gap


def test_markdown_surfaces_beginner_friendly_risk_labels():
    artifact = CalibrationArtifact(
        method="p1",
        unlock_k=1,
        selected_profile=ExecutionProfile.EXPEDITION,
        neutral_baseline_params={},
        calibrated_params={},
        neutral_fitness=0.4,
        calibrated_fitness=0.5,
        uplift_absolute=0.1,
        uplift_percent=25.0,
        quality_per_cost_neutral=0.01,
        quality_per_cost_best=0.02,
        quality_per_latency_neutral=0.01,
        quality_per_latency_best=0.02,
        best_params={},
        best_fitness=0.5,
        hard_gate_pass_rate=1.0,
        n_candidates_evaluated=1,
        total_api_calls=2,
        boundary_warnings=[
            {
                "code": "weak_judge",
                "category": "validation strength",
                "severity": "warning",
                "summary": "Judge is not ship-grade for held-out validation.",
                "detail": "Use a stronger cloud judge.",
            }
        ],
        ship_recommendation="experiment",
        stayed_within_guarded_boundaries=False,
        additional_uplift_from_boundary_crossing=0.05,
    )
    rendered = render_markdown(artifact)
    assert "validation strength" in rendered
    assert "Boundary Warnings" in rendered
    assert "Boundary-crossing uplift" in rendered


# ---------------------------------------------------------------------------
# Reviewer P1 #15: central profile-policy enforcement.
# ``enforce_profile_policy`` consolidates the policy slots that key off
# provider capabilities so runtime/CLI/preflight all see the same gate.
# ---------------------------------------------------------------------------


def test_enforce_profile_policy_blocks_experimental_target_under_guarded():
    """The gap that motivated this function: assess_run_risk emits
    weak_judge for non-ship-grade/experimental judges, but does not
    block an experimental *target* under guarded. ``enforce_profile_policy``
    fills only that gap (no duplicate weak_judge emission)."""
    from omegaprompt.core.profiles import enforce_profile_policy

    target = _caps(
        provider="local-x",
        tier=CapabilityTier.LOCAL,
        ship_grade_judge=False,
        experimental=True,
    )
    judge = _caps(
        provider="anthropic", tier=CapabilityTier.CLOUD, ship_grade_judge=True
    )
    warnings = enforce_profile_policy(ExecutionProfile.GUARDED, target, judge)
    codes = {w.code for w in warnings}
    assert "experimental_target_under_guarded" in codes
    assert all(w.severity == "critical" for w in warnings)


def test_enforce_profile_policy_does_not_duplicate_weak_judge():
    """``assess_run_risk`` already covers experimental / non-ship-grade
    *judges* via ``weak_judge``. ``enforce_profile_policy`` must not
    emit duplicate warnings for the same fact, otherwise the artifact's
    boundary_warnings list grows noisy."""
    from omegaprompt.core.profiles import enforce_profile_policy

    target = _caps(
        provider="anthropic", tier=CapabilityTier.CLOUD, ship_grade_judge=True
    )
    judge = _caps(
        provider="local-x",
        tier=CapabilityTier.LOCAL,
        ship_grade_judge=False,
        experimental=True,
    )
    warnings = enforce_profile_policy(ExecutionProfile.GUARDED, target, judge)
    assert warnings == []  # weak_judge handled elsewhere


def test_enforce_profile_policy_silent_under_expedition():
    """Expedition profile sets allow_experimental_providers=True; the
    central enforcer must return no warnings for inputs that fail
    under guarded."""
    from omegaprompt.core.profiles import enforce_profile_policy

    target = _caps(
        provider="local-x",
        tier=CapabilityTier.LOCAL,
        ship_grade_judge=False,
        experimental=True,
    )
    judge = _caps(
        provider="local-x",
        tier=CapabilityTier.LOCAL,
        ship_grade_judge=False,
        experimental=True,
    )
    warnings = enforce_profile_policy(ExecutionProfile.EXPEDITION, target, judge)
    assert warnings == []


def test_enforce_profile_policy_clean_pair_passes_guarded():
    from omegaprompt.core.profiles import enforce_profile_policy

    target = _caps(
        provider="anthropic", tier=CapabilityTier.CLOUD, ship_grade_judge=True
    )
    judge = _caps(
        provider="anthropic", tier=CapabilityTier.CLOUD, ship_grade_judge=True
    )
    warnings = enforce_profile_policy(ExecutionProfile.GUARDED, target, judge)
    assert warnings == []


# ---------------------------------------------------------------------------
# Reviewer P2 #18: ArtifactStatus enum closes the open-string drift that
# let REQUIRES_MANUAL_REVIEW exist in the adaptation path while the
# field's description didn't list it.
# ---------------------------------------------------------------------------


def test_artifact_status_enum_accepts_documented_values():
    from omegaprompt.domain.result import ArtifactStatus

    artifact = CalibrationArtifact(
        method="p1",
        unlock_k=1,
        best_params={"system_prompt_variant": 0},
        best_fitness=0.5,
        hard_gate_pass_rate=1.0,
        n_candidates_evaluated=1,
        total_api_calls=1,
        status=ArtifactStatus.REQUIRES_MANUAL_REVIEW,
    )
    assert artifact.status == ArtifactStatus.REQUIRES_MANUAL_REVIEW
    # JSON round-trip preserves the enum value as its string form.
    rt = CalibrationArtifact.model_validate_json(artifact.model_dump_json())
    assert rt.status == ArtifactStatus.REQUIRES_MANUAL_REVIEW


def test_artifact_status_enum_rejects_unknown_strings():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CalibrationArtifact(
            method="p1",
            unlock_k=1,
            best_params={"system_prompt_variant": 0},
            best_fitness=0.5,
            hard_gate_pass_rate=1.0,
            n_candidates_evaluated=1,
            total_api_calls=1,
            status="UNDOCUMENTED_STATUS",  # not in the enum
        )


def test_runtime_calibrate_emits_experimental_target_warning_under_guarded(monkeypatch):
    """End-to-end drift guard for Reviewer P1 #15: a future refactor that
    drops the ``boundary_warnings.extend(enforce_profile_policy(...))``
    call from runtime.calibrate would silently regress the policy. This
    test fails-closed against that — the warning must reach the
    artifact through the public ``calibrate`` entrypoint."""
    from omegaprompt import runtime as rt
    from omegaprompt.providers.base import (
        CapabilityTier,
        ProviderCapabilities,
        ProviderResponse,
    )
    from omegaprompt.domain.judge import (
        Dimension,
        HardGate,
        JudgeResult,
        JudgeRubric,
    )
    from omegaprompt.domain.params import PromptVariants
    from tests.helpers import workspace_tmpdir

    class ExperimentalProvider:
        name = "experimental-target"
        model = "x"

        def capabilities(self):
            return ProviderCapabilities(
                provider=self.name,
                tier=CapabilityTier.LOCAL,
                supports_strict_schema=False,
                supports_json_object=True,
                supports_reasoning_profiles=False,
                supports_usage_accounting=True,
                supports_llm_judge=False,
                ship_grade_judge=False,
                experimental=True,  # the gate the policy fires on
            )

        def call(self, request):
            return ProviderResponse(
                text="ok",
                usage={
                    "input_tokens": 5,
                    "output_tokens": 3,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            )

    class ShipGradeJudgeProvider:
        name = "ship-judge"
        model = "y"

        def capabilities(self):
            return ProviderCapabilities(
                provider=self.name,
                tier=CapabilityTier.CLOUD,
                supports_strict_schema=True,
                supports_json_object=True,
                supports_reasoning_profiles=True,
                supports_usage_accounting=True,
                supports_llm_judge=True,
                ship_grade_judge=True,
            )

        def call(self, request):
            return ProviderResponse(
                parsed=JudgeResult(scores={"q": 1}, gate_results={"g": True}),
                usage={
                    "input_tokens": 5,
                    "output_tokens": 3,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            )

    target = ExperimentalProvider()
    judge = ShipGradeJudgeProvider()

    rubric = JudgeRubric(
        dimensions=[Dimension(name="q", description="q", weight=1.0)],
        hard_gates=[HardGate(name="g", description="g", evaluator="judge")],
    )
    variants = PromptVariants(system_prompts=["sp-A"])

    with workspace_tmpdir() as tmp:
        train_path = tmp / "train.jsonl"
        train_path.write_text(
            '{"id":"t1","input":"a"}\n{"id":"t2","input":"b"}\n',
            encoding="utf-8",
        )
        artifact = rt.calibrate(
            train=train_path,
            rubric=rubric,
            variants=variants,
            target=target,
            judge=judge,
            tuning=rt.CalibrateTuning(
                method="p1",
                unlock_k=1,
                profile=ExecutionProfile.GUARDED,
            ),
        )

    codes = {w.code for w in artifact.boundary_warnings}
    assert "experimental_target_under_guarded" in codes


def test_artifact_status_enum_accepts_string_form_for_documented_values():
    """String round-trips (artifact JSON files) still load: pydantic
    coerces a string that matches an enum value to the enum member."""
    from omegaprompt.domain.result import ArtifactStatus

    artifact = CalibrationArtifact(
        method="p1",
        unlock_k=1,
        best_params={"system_prompt_variant": 0},
        best_fitness=0.5,
        hard_gate_pass_rate=1.0,
        n_candidates_evaluated=1,
        total_api_calls=1,
        status="OK",  # plain str
    )
    assert artifact.status == ArtifactStatus.OK
