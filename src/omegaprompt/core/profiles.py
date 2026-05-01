"""Execution profile policies."""

from __future__ import annotations

from dataclasses import dataclass

from omegaprompt.domain.profiles import (
    BoundaryWarning,
    ExecutionProfile,
    RelaxedSafeguard,
    RiskCategory,
)


@dataclass(frozen=True)
class ProfilePolicy:
    profile: ExecutionProfile
    allow_schema_degradation: bool
    allow_experimental_providers: bool
    allow_non_ship_grade_judge: bool
    default_max_gap: float
    default_min_kc4: float


_POLICIES = {
    ExecutionProfile.GUARDED: ProfilePolicy(
        profile=ExecutionProfile.GUARDED,
        allow_schema_degradation=False,
        allow_experimental_providers=False,
        allow_non_ship_grade_judge=False,
        default_max_gap=0.25,
        default_min_kc4=0.5,
    ),
    ExecutionProfile.EXPEDITION: ProfilePolicy(
        profile=ExecutionProfile.EXPEDITION,
        allow_schema_degradation=True,
        allow_experimental_providers=True,
        allow_non_ship_grade_judge=True,
        default_max_gap=0.35,
        default_min_kc4=0.3,
    ),
}


def policy_for(profile: ExecutionProfile) -> ProfilePolicy:
    return _POLICIES[profile]


def enforce_profile_policy(
    profile: ExecutionProfile,
    target_capabilities,  # ProviderCapabilities; typed loosely to avoid cycle
    judge_capabilities,   # ProviderCapabilities
) -> list[BoundaryWarning]:
    """Single-source check for profile-policy slots not already covered.

    Reviewer P1 #15: ``ProfilePolicy`` declares
    ``allow_schema_degradation``, ``allow_experimental_providers``,
    ``allow_non_ship_grade_judge``. The rules were enforced ad-hoc
    across ``LLMJudge``, the local provider's strict-schema fallback,
    and ``assess_run_risk``. ``runtime.calibrate`` /
    ``commands.calibrate`` / ``preflight`` were free to drift.

    Coverage matrix (what fires where):

    - ``placeholder`` provider — ``assess_run_risk`` (critical).
    - ``judge`` non-ship-grade or experimental — ``assess_run_risk``
      emits ``weak_judge``.
    - ``target`` experimental under guarded — *gap* before this fix.
      Now emitted here as ``experimental_target_under_guarded``.
    - ``schema_degradation`` — checked at the adapter call site, since
      the fallback is a runtime event, not a static capability fact.

    Returning only the gap (no duplicates of ``weak_judge``) keeps
    ``boundary_warnings`` from doubling up on the same fact. Callers
    append the returned list to their aggregate warnings; empty means
    the policy passed.
    """
    policy = policy_for(profile)
    warnings: list[BoundaryWarning] = []

    if not policy.allow_experimental_providers:
        if target_capabilities.experimental:
            warnings.append(
                BoundaryWarning(
                    code="experimental_target_under_guarded",
                    category=RiskCategory.EXPERIMENTAL_RISK,
                    severity="critical",
                    summary=(
                        "Experimental target provider blocked under "
                        f"{profile.value} profile."
                    ),
                    detail=(
                        f"Target provider {target_capabilities.provider!r} "
                        "declares experimental=True; the "
                        f"{profile.value} profile forbids experimental "
                        "providers. Switch to a ship-grade target or run "
                        "under expedition profile."
                    ),
                )
            )

    return warnings


def relaxed_safeguards_for(profile: ExecutionProfile) -> list[RelaxedSafeguard]:
    if profile != ExecutionProfile.EXPEDITION:
        return []
    return [
        RelaxedSafeguard(
            name="schema_fallbacks",
            reason="Allow adapters to degrade from strict schema to JSON validation when native support is absent.",
            increased_risk="Validation strength falls below native strict parsing.",
        ),
        RelaxedSafeguard(
            name="experimental_providers",
            reason="Allow experimental or local adapters when the expected efficiency-adjusted uplift is large.",
            increased_risk="Capability gaps may reduce deployment readiness or judge reliability.",
        ),
    ]
