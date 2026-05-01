"""Reviewer P4: four latent runtime/MCP bugs.

Each test reproduces the original failure and pins the fix:

1. grade() previously returned the raw tuple from Judge.score(), so
   MCP's `result.model_dump()` crashed with AttributeError.
2. _resolve_params(CalibrationArtifact) called `.model_dump()` on a
   plain dict (calibrated_params is dict[str, Any], not a BaseModel),
   which crashed any caller that passed an artifact for evaluate().
3. preflight() referenced ``is_placeholder`` / ``is_experimental``
   attributes that don't exist on ProviderCapabilities — the canonical
   field names are ``placeholder`` / ``experimental``. Real adapter
   capabilities (e.g. Gemini placeholder) silently passed the guard.
4. preflight() now also surfaces an experimental-judge warning, not
   just experimental-target.
"""

from __future__ import annotations

import pytest

from omegaprompt.domain import ShipRecommendation
from omegaprompt.domain.dataset import DatasetItem
from omegaprompt.domain.judge import Dimension, HardGate, JudgeRubric
from omegaprompt.domain.result import CalibrationArtifact
from omegaprompt.preflight.contracts import PreflightStatus
from omegaprompt.providers.base import CapabilityTier, ProviderCapabilities
from omegaprompt.runtime import GradeResult, _resolve_params, evaluate, grade, preflight


def _rubric_rule_only() -> JudgeRubric:
    return JudgeRubric(
        dimensions=[Dimension(name="acc", description="accuracy", weight=1.0)],
        hard_gates=[
            HardGate(name="no_refusal", description="no refusal", evaluator="rule"),
        ],
    )


# ---------------------------------------------------------------------------
# Bug #1: grade() must return a serialisable wrapper, not a tuple.
# ---------------------------------------------------------------------------


def test_grade_returns_grade_result_not_tuple():
    """Regression: grade() used to return tuple[JudgeResult, dict]."""
    result = grade(
        rubric=_rubric_rule_only(),
        item=DatasetItem(id="t1", input="hello", reference=None),
        response="A real and substantive response that the judge can score.",
        provider="anthropic",
        strategy="rule",
    )
    assert isinstance(result, GradeResult)
    # Tuples don't have model_dump; this would have crashed pre-fix.
    payload = result.model_dump(mode="json")
    assert payload["judge"]["gate_results"]["no_refusal"] is True
    assert isinstance(payload["usage"], dict)


def test_grade_result_carries_usage_dict():
    result = grade(
        rubric=_rubric_rule_only(),
        item={"id": "t1", "input": "hi"},
        response="A real attempt at the input.",
        provider="anthropic",
        strategy="rule",
    )
    # Rule-judge usage is zero-valued but the keys are present.
    assert "input_tokens" in result.usage
    assert result.usage["input_tokens"] == 0


# ---------------------------------------------------------------------------
# Bug #2: _resolve_params(CalibrationArtifact) must not crash on dict.
# ---------------------------------------------------------------------------


def test_resolve_params_accepts_calibration_artifact():
    """Regression: _resolve_params called .model_dump() on a plain dict."""
    artifact = CalibrationArtifact(
        method="p1",
        unlock_k=3,
        best_params={"system_prompt_variant": 2},
        best_fitness=0.8,
        calibrated_params={"system_prompt_variant": 2, "few_shot_count": 1},
        calibrated_fitness=0.8,
        hard_gate_pass_rate=1.0,
        n_candidates_evaluated=1,
        total_api_calls=1,
        ship_recommendation=ShipRecommendation.SHIP,
    )
    out = _resolve_params(artifact)
    assert isinstance(out, dict)
    assert out["system_prompt_variant"] == 2
    assert out["few_shot_count"] == 1
    # Defensive: the returned dict must be a copy, not the artifact's
    # internal dict, so callers can't mutate the artifact.
    out["system_prompt_variant"] = 99
    assert artifact.calibrated_params["system_prompt_variant"] == 2


def test_resolve_params_passes_through_dict():
    out = _resolve_params({"a": 1})
    assert out == {"a": 1}


# ---------------------------------------------------------------------------
# Bug #3: preflight() must read ``placeholder`` / ``experimental``, not
#         ``is_placeholder`` / ``is_experimental``.
# ---------------------------------------------------------------------------


def _patch_provider(monkeypatch, target_caps, judge_caps):
    """Monkeypatch make_provider + provider_capabilities for a preflight call."""
    class StubProvider:
        def __init__(self, name):
            self.name = name
            self.model = "stub"

    monkeypatch.setattr(
        "omegaprompt.runtime.make_provider",
        lambda n, model=None, base_url=None: StubProvider(n),
    )
    caps_map = {"target_provider": target_caps, "judge_provider": judge_caps}
    # provider_capabilities is keyed off the provider object; we only
    # have two providers so a dispatch by name suffices.
    name_to_caps = {"openai": target_caps, "anthropic": judge_caps}
    monkeypatch.setattr(
        "omegaprompt.runtime.provider_capabilities",
        lambda p: name_to_caps[p.name],
    )
    return caps_map


def test_preflight_blocks_real_placeholder_target_under_guarded(monkeypatch):
    """Pre-fix: real ProviderCapabilities(placeholder=True) slipped through."""
    target_caps = ProviderCapabilities(
        provider="openai",
        tier=CapabilityTier.CLOUD,
        placeholder=True,
    )
    judge_caps = ProviderCapabilities(
        provider="anthropic",
        tier=CapabilityTier.CLOUD,
    )
    _patch_provider(monkeypatch, target_caps, judge_caps)

    report = preflight(target="openai", judge="anthropic", profile="guarded")
    assert report.status == PreflightStatus.ABORT
    assert any("placeholder" in r for r in report.blocker_reasons)


def test_preflight_blocks_real_placeholder_judge_under_guarded(monkeypatch):
    target_caps = ProviderCapabilities(
        provider="openai",
        tier=CapabilityTier.CLOUD,
    )
    judge_caps = ProviderCapabilities(
        provider="anthropic",
        tier=CapabilityTier.CLOUD,
        placeholder=True,
    )
    _patch_provider(monkeypatch, target_caps, judge_caps)

    report = preflight(target="openai", judge="anthropic", profile="guarded")
    assert report.status == PreflightStatus.ABORT
    assert any("judge" in r and "placeholder" in r for r in report.blocker_reasons)


def test_preflight_blocks_experimental_target_under_guarded(monkeypatch):
    """Reviewer P1 #15: experimental target under guarded is a blocker,
    not a soft warning. Pre-fix preflight returned PROCEED with a
    warning string; post-fix it routes through enforce_profile_policy
    and surfaces a critical blocker."""
    target_caps = ProviderCapabilities(
        provider="openai",
        tier=CapabilityTier.CLOUD,
        experimental=True,
    )
    judge_caps = ProviderCapabilities(
        provider="anthropic",
        tier=CapabilityTier.CLOUD,
    )
    _patch_provider(monkeypatch, target_caps, judge_caps)

    report = preflight(target="openai", judge="anthropic", profile="guarded")
    assert report.status == PreflightStatus.ABORT
    assert any("Experimental target" in r for r in report.blocker_reasons)


def test_preflight_warns_on_experimental_target_under_expedition(monkeypatch):
    """Under expedition profile the policy allows experimental providers,
    so the same input that aborts under guarded proceeds with a soft
    warning instead."""
    target_caps = ProviderCapabilities(
        provider="openai",
        tier=CapabilityTier.CLOUD,
        experimental=True,
    )
    judge_caps = ProviderCapabilities(
        provider="anthropic",
        tier=CapabilityTier.CLOUD,
    )
    _patch_provider(monkeypatch, target_caps, judge_caps)

    report = preflight(target="openai", judge="anthropic", profile="expedition")
    assert report.status == PreflightStatus.PROCEED


def test_preflight_warns_on_experimental_judge(monkeypatch):
    target_caps = ProviderCapabilities(
        provider="openai",
        tier=CapabilityTier.CLOUD,
    )
    judge_caps = ProviderCapabilities(
        provider="anthropic",
        tier=CapabilityTier.CLOUD,
        experimental=True,
    )
    _patch_provider(monkeypatch, target_caps, judge_caps)

    report = preflight(target="openai", judge="anthropic", profile="guarded")
    assert any("judge" in w and "experimental" in w for w in report.warnings)


def test_preflight_clean_caps_proceeds(monkeypatch):
    target_caps = ProviderCapabilities(
        provider="openai", tier=CapabilityTier.CLOUD,
    )
    judge_caps = ProviderCapabilities(
        provider="anthropic", tier=CapabilityTier.CLOUD,
    )
    _patch_provider(monkeypatch, target_caps, judge_caps)

    report = preflight(target="openai", judge="anthropic", profile="guarded")
    assert report.status == PreflightStatus.PROCEED
    assert report.blocker_reasons == []
