"""Opt-in live provider smoke tests.

Default pytest and default CI must never call provider APIs. These tests require
both OMEGAPROMPT_LIVE_PROVIDER_TESTS=1 and an explicit OMEGAPROMPT_LIVE_PROVIDER
selection before any provider call is attempted.
"""

from __future__ import annotations

import os

import pytest

from omegaprompt.domain.enums import OutputBudgetBucket, ReasoningProfile, ResponseSchemaMode
from omegaprompt.domain.profiles import ExecutionProfile
from omegaprompt.providers import make_provider
from omegaprompt.providers.base import ProviderRequest


pytestmark = pytest.mark.live


@pytest.mark.skipif(
    os.getenv("OMEGAPROMPT_LIVE_PROVIDER_TESTS") != "1",
    reason="Set OMEGAPROMPT_LIVE_PROVIDER_TESTS=1 to opt into live provider smoke tests.",
)
def test_live_provider_freeform_smoke_requires_explicit_provider_and_key() -> None:
    provider_name = os.getenv("OMEGAPROMPT_LIVE_PROVIDER")
    if not provider_name:
        pytest.skip("Set OMEGAPROMPT_LIVE_PROVIDER to choose a live provider.")

    required_env_by_provider = {
        "anthropic": ("ANTHROPIC_API_KEY",),
        "openai": ("OPENAI_API_KEY",),
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    }
    if provider_name not in required_env_by_provider:
        pytest.skip("Live smoke is limited to anthropic, openai, or gemini.")

    required_env = required_env_by_provider[provider_name]
    if not any(os.getenv(name) for name in required_env):
        pytest.skip(f"Set one of {', '.join(required_env)} for {provider_name}.")

    provider = make_provider(provider_name)
    response = provider.call(
        ProviderRequest(
            system_prompt="Reply with exactly: ok",
            user_message="ok",
            reasoning_profile=ReasoningProfile.OFF,
            output_budget_bucket=OutputBudgetBucket.SHORT,
            response_schema_mode=ResponseSchemaMode.FREEFORM,
            execution_profile=ExecutionProfile.GUARDED,
        )
    )

    assert response.text.strip()
