"""Legacy API shim tests - the v0.2 call_target / effort helpers.

The v1.0 canonical path is ``provider.call(ProviderRequest(...))``; see
:mod:`tests.test_providers` for coverage of the new contract. These
tests only verify the legacy shim in :mod:`omegaprompt.api` keeps
translating v0.2 kwargs into v1.0 ``ProviderRequest`` correctly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from omegaprompt.api import (
    _EFFORT_LABELS,
    _MAX_TOKENS_BUCKETS,
    call_target,
    effort_from_int,
    max_tokens_from_int,
)
from omegaprompt.domain.enums import OutputBudgetBucket, ReasoningProfile
from omegaprompt.providers.base import ProviderResponse


def test_effort_from_int_clamps():
    assert effort_from_int(-99) == _EFFORT_LABELS[0]
    assert effort_from_int(0) == "low"
    assert effort_from_int(1) == "medium"
    assert effort_from_int(2) == "high"
    assert effort_from_int(99) == _EFFORT_LABELS[-1]


def test_max_tokens_from_int_clamps():
    assert max_tokens_from_int(-5) == _MAX_TOKENS_BUCKETS[0]
    assert max_tokens_from_int(0) == 1024
    assert max_tokens_from_int(99) == _MAX_TOKENS_BUCKETS[-1]


def test_call_target_translates_to_provider_request():
    provider = MagicMock()
    provider.name = "mock"
    provider.model = "m"
    provider.call.return_value = ProviderResponse(
        text="response text",
        usage={"input_tokens": 10, "output_tokens": 5},
    )

    text, usage = call_target(
        provider,
        system_prompt="SP",
        user_message="UM",
        few_shots=[{"input": "a", "output": "b"}],
        effort="medium",
        max_tokens=4096,
        thinking_enabled=True,
    )

    assert text == "response text"
    assert usage["input_tokens"] == 10
    request = provider.call.call_args.args[0]
    assert request.system_prompt == "SP"
    assert request.user_message == "UM"
    assert request.few_shots == [{"input": "a", "output": "b"}]
    assert request.reasoning_profile == ReasoningProfile.STANDARD
    assert request.output_budget == OutputBudgetBucket.MEDIUM


def test_call_target_thinking_disabled_maps_to_off_profile():
    provider = MagicMock()
    provider.name = "mock"
    provider.model = "m"
    provider.call.return_value = ProviderResponse(text="ok", usage={})

    call_target(
        provider,
        system_prompt="SP",
        user_message="U",
        few_shots=[],
        effort="low",
        max_tokens=1024,
        thinking_enabled=False,
    )
    request = provider.call.call_args.args[0]
    assert request.reasoning_profile == ReasoningProfile.OFF
    assert request.output_budget == OutputBudgetBucket.SMALL


def test_call_target_high_effort_large_budget_maps_to_deep_large():
    provider = MagicMock()
    provider.name = "mock"
    provider.model = "m"
    provider.call.return_value = ProviderResponse(text="ok", usage={})

    call_target(
        provider,
        system_prompt="SP",
        user_message="U",
        few_shots=[],
        effort="high",
        max_tokens=16000,
        thinking_enabled=True,
    )
    request = provider.call.call_args.args[0]
    assert request.reasoning_profile == ReasoningProfile.DEEP
    assert request.output_budget == OutputBudgetBucket.LARGE
