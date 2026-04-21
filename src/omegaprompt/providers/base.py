"""Unified ``LLMProvider`` Protocol + provider-neutral request/response.

v1.0 change: the two-method interface (``complete`` + ``structured_complete``)
is replaced by a single ``call()`` that consumes a ``ProviderRequest``
carrying the meta-axes. Each adapter translates the meta-axes to its
vendor's native parameters internally; the public contract is vendor-free.

Returning a :class:`ProviderResponse` rather than a tuple keeps call sites
readable when response_schema_mode varies at runtime.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from omegaprompt.domain.enums import (
    OutputBudgetBucket,
    ReasoningProfile,
    ResponseSchemaMode,
    ToolPolicyVariant,
)

T = TypeVar("T", bound=BaseModel)


class ProviderError(RuntimeError):
    """Raised when a provider call fails in a way the caller can surface."""


class ProviderRequest(BaseModel):
    """The provider-neutral call payload.

    Adapters see only this shape. Vendor-specific knobs (Anthropic
    ``thinking`` config, OpenAI ``response_format``) are derived from the
    meta-axes inside each adapter's ``call``.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    system_prompt: str
    user_message: str
    few_shots: list[dict[str, str]] = Field(default_factory=list)

    reasoning_profile: ReasoningProfile = ReasoningProfile.STANDARD
    output_budget: OutputBudgetBucket = OutputBudgetBucket.MEDIUM
    response_schema_mode: ResponseSchemaMode = ResponseSchemaMode.FREEFORM
    tool_policy: ToolPolicyVariant = ToolPolicyVariant.NO_TOOLS

    # Required when response_schema_mode == STRICT_SCHEMA. Otherwise ignored.
    output_schema: type[BaseModel] | None = None


class ProviderResponse(BaseModel):
    """The provider-neutral response."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    text: str = ""
    parsed: BaseModel | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    finish_reason: str | None = None


class LLMProvider(Protocol):
    """Structural interface for LLM adapters."""

    name: str
    model: str

    def call(self, request: ProviderRequest) -> ProviderResponse: ...


def empty_usage() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def normalize_usage(raw: Any) -> dict[str, int]:
    """Coerce vendor usage into the canonical dict shape."""
    def _attr(name: str) -> int:
        value = getattr(raw, name, 0) if raw is not None else 0
        return int(value or 0)

    if isinstance(raw, dict):
        def _attr(name: str) -> int:  # noqa: F811
            return int(raw.get(name, 0) or 0)

    input_tokens = _attr("input_tokens") or _attr("prompt_tokens")
    output_tokens = _attr("output_tokens") or _attr("completion_tokens")
    cache_create = _attr("cache_creation_input_tokens")
    cache_read = _attr("cache_read_input_tokens")

    if not cache_read and raw is not None:
        details = getattr(raw, "prompt_tokens_details", None)
        if details is None and isinstance(raw, dict):
            details = raw.get("prompt_tokens_details")
        if details is not None:
            if isinstance(details, dict):
                cache_read = int(details.get("cached_tokens", 0) or 0)
            else:
                cache_read = int(getattr(details, "cached_tokens", 0) or 0)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_create,
        "cache_read_input_tokens": cache_read,
    }


# ----- Meta-axis to vendor-param translation (shared helpers) -----

_OUTPUT_BUDGET_MAX_TOKENS: dict[OutputBudgetBucket, int] = {
    OutputBudgetBucket.SMALL: 1024,
    OutputBudgetBucket.MEDIUM: 4096,
    OutputBudgetBucket.LARGE: 16000,
}

_REASONING_EFFORT_LABEL: dict[ReasoningProfile, str] = {
    ReasoningProfile.OFF: "low",
    ReasoningProfile.LIGHT: "low",
    ReasoningProfile.STANDARD: "medium",
    ReasoningProfile.DEEP: "high",
}


def max_tokens_for(budget: OutputBudgetBucket) -> int:
    return _OUTPUT_BUDGET_MAX_TOKENS[budget]


def reasoning_effort_label(profile: ReasoningProfile) -> str:
    return _REASONING_EFFORT_LABEL[profile]


def reasoning_enabled(profile: ReasoningProfile) -> bool:
    return profile != ReasoningProfile.OFF
