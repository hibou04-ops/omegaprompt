"""Backward-compat shim - the v0.2 API boundary helpers.

v1.0 folds the ``call_target`` / ``call_judge`` functions into the
:class:`omegaprompt.targets.PromptTarget` + :class:`omegaprompt.judges.Judge`
layer. The v0.2 helpers below are thin legacy adapters for any downstream
code that imported them directly.

New code should:

- Call target via ``provider.call(ProviderRequest(...))``.
- Score via ``judge.score(rubric=..., item=..., target_response=...)``.
"""

from __future__ import annotations

from omegaprompt.domain.enums import (
    OutputBudgetBucket,
    ReasoningProfile,
    ResponseSchemaMode,
)
from omegaprompt.providers.base import LLMProvider, ProviderRequest


# Legacy integer-index effort/max_tokens helpers.
_EFFORT_LABELS = ("low", "medium", "high")
_MAX_TOKENS_BUCKETS = (1024, 4096, 16000)


def effort_from_int(idx: int) -> str:
    idx = max(0, min(len(_EFFORT_LABELS) - 1, idx))
    return _EFFORT_LABELS[idx]


def max_tokens_from_int(idx: int) -> int:
    idx = max(0, min(len(_MAX_TOKENS_BUCKETS) - 1, idx))
    return _MAX_TOKENS_BUCKETS[idx]


def call_target(
    provider: LLMProvider,
    *,
    system_prompt: str,
    user_message: str,
    few_shots: list[dict[str, str]],
    effort: str = "medium",
    max_tokens: int = 4096,
    thinking_enabled: bool = True,
) -> tuple[str, dict[str, int]]:
    """Legacy free-form target call. Prefer ``provider.call(ProviderRequest(...))``."""
    profile = _legacy_reasoning_profile(effort, thinking_enabled)
    budget = _legacy_output_budget(max_tokens)
    request = ProviderRequest(
        system_prompt=system_prompt,
        user_message=user_message,
        few_shots=few_shots,
        reasoning_profile=profile,
        output_budget=budget,
        response_schema_mode=ResponseSchemaMode.FREEFORM,
    )
    response = provider.call(request)
    return response.text, response.usage


def _legacy_reasoning_profile(effort: str, thinking_enabled: bool) -> ReasoningProfile:
    if not thinking_enabled:
        return ReasoningProfile.OFF
    return {
        "low": ReasoningProfile.LIGHT,
        "medium": ReasoningProfile.STANDARD,
        "high": ReasoningProfile.DEEP,
    }.get(effort.lower(), ReasoningProfile.STANDARD)


def _legacy_output_budget(max_tokens: int) -> OutputBudgetBucket:
    if max_tokens <= 1500:
        return OutputBudgetBucket.SMALL
    if max_tokens <= 6000:
        return OutputBudgetBucket.MEDIUM
    return OutputBudgetBucket.LARGE


__all__ = [
    "effort_from_int",
    "max_tokens_from_int",
    "call_target",
]
