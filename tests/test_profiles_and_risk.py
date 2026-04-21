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
