"""Execution profile policies."""

from __future__ import annotations

from dataclasses import dataclass

from omegaprompt.domain.profiles import ExecutionProfile, RelaxedSafeguard


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
