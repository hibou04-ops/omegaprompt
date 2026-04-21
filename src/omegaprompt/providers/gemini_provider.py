"""Gemini adapter placeholder with explicit capability notes."""

from __future__ import annotations

from omegaprompt.providers.base import (
    CapabilityTier,
    ProviderCapabilities,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
)


class GeminiProvider:
    """Clean placeholder until a full native Gemini adapter lands."""

    name = "gemini"

    def __init__(self, *, model: str, **_: object) -> None:
        self.model = model

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider=self.name,
            tier=CapabilityTier.CLOUD,
            supports_strict_schema=False,
            supports_json_object=False,
            supports_reasoning_profiles=False,
            supports_usage_accounting=False,
            supports_llm_judge=False,
            ship_grade_judge=False,
            supports_tools=False,
            experimental=True,
            placeholder=True,
            notes=[
                "Placeholder only in this refactor.",
                "Use Anthropic or OpenAI for ship-grade judge paths until a native Gemini adapter is added.",
            ],
        )

    def call(self, request: ProviderRequest) -> ProviderResponse:  # pragma: no cover - explicit placeholder
        raise ProviderError(
            "Gemini is wired as a placeholder in this refactor. "
            "The core remains provider-neutral, but the native Gemini adapter is not implemented yet."
        )
