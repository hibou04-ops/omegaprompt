"""Structural risk reporting for profiles, providers, and validation."""

from __future__ import annotations

from omegaprompt.domain.profiles import (
    BoundaryWarning,
    ExecutionProfile,
    RiskCategory,
    ShipRecommendation,
)
from omegaprompt.providers.base import CapabilityEvent, ProviderCapabilities


def assess_run_risk(
    *,
    profile: ExecutionProfile,
    target_capabilities: ProviderCapabilities,
    judge_capabilities: ProviderCapabilities,
    degraded_capabilities: list[CapabilityEvent],
    has_walk_forward: bool,
    walk_forward_passed: bool | None,
) -> tuple[list[BoundaryWarning], bool, ShipRecommendation]:
    warnings: list[BoundaryWarning] = []

    if target_capabilities.placeholder:
        warnings.append(
            BoundaryWarning(
                code="placeholder_provider",
                category=RiskCategory.EXPERIMENTAL_RISK,
                severity="critical",
                summary="Placeholder provider detected.",
                detail=(
                    f"{target_capabilities.provider} is only a placeholder adapter in this release."
                ),
            )
        )

    if judge_capabilities.experimental or not judge_capabilities.ship_grade_judge:
        warnings.append(
            BoundaryWarning(
                code="weak_judge",
                category=RiskCategory.VALIDATION_STRENGTH,
                severity="critical" if profile == ExecutionProfile.GUARDED else "warning",
                summary="Judge is not ship-grade for held-out validation.",
                detail=(
                    "Use a stronger cloud judge for ship decisions. Rule-first local judging "
                    "is acceptable for exploration, not for final deployment gates."
                ),
            )
        )

    if not has_walk_forward:
        warnings.append(
            BoundaryWarning(
                code="no_walk_forward",
                category=RiskCategory.DEPLOYMENT_READINESS,
                severity="critical",
                summary="Held-out validation is missing.",
                detail="Deployment readiness is unknown without walk-forward evaluation.",
            )
        )
    elif walk_forward_passed is False:
        warnings.append(
            BoundaryWarning(
                code="walk_forward_failed",
                category=RiskCategory.DEPLOYMENT_READINESS,
                severity="critical",
                summary="Held-out validation failed.",
                detail="The train-best candidate did not clear the walk-forward ship gate.",
            )
        )

    if degraded_capabilities:
        warnings.append(
            BoundaryWarning(
                code="capability_fallbacks",
                category=RiskCategory.SAFETY_BOUNDARY,
                severity="warning",
                summary="Requested features were degraded by an adapter fallback.",
                detail="Validation strength or control fidelity was lowered to keep the run moving.",
            )
        )

    within_guarded = all(not w.affects_guarded_boundary for w in warnings) or all(
        w.severity == "info" for w in warnings
    )
    if profile == ExecutionProfile.GUARDED:
        within_guarded = within_guarded and not degraded_capabilities
        within_guarded = within_guarded and judge_capabilities.ship_grade_judge
        within_guarded = within_guarded and has_walk_forward and walk_forward_passed is not False

    ship_recommendation = ShipRecommendation.SHIP
    if any(w.severity == "critical" for w in warnings):
        ship_recommendation = ShipRecommendation.BLOCK
    elif warnings:
        ship_recommendation = (
            ShipRecommendation.HOLD
            if profile == ExecutionProfile.GUARDED
            else ShipRecommendation.EXPERIMENT
        )

    return warnings, within_guarded, ship_recommendation
