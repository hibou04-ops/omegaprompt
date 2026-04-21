"""Provider factory for omegaprompt.

Same pattern as antemortem-cli: one call site to construct the right
adapter from a string name. Extending: implement ``LLMProvider`` in a
new module, register in ``_REGISTRY``.
"""

from __future__ import annotations

from typing import Any

from omegaprompt.providers.base import LLMProvider, ProviderError


DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-4-7",
    "openai": "gpt-4o",
}


def supported_providers() -> list[str]:
    return list(_REGISTRY.keys())


def make_provider(
    name: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    client: Any = None,
    **extra: Any,
) -> LLMProvider:
    """Construct a configured ``LLMProvider`` by name."""
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise ProviderError(
            f"Unknown provider {name!r}. Supported: {', '.join(supported_providers())}"
        )

    builder = _REGISTRY[key]
    resolved_model = model or DEFAULT_MODELS[key]
    return builder(
        model=resolved_model,
        api_key=api_key,
        base_url=base_url,
        client=client,
        **extra,
    )


def _build_anthropic(**kwargs: Any) -> LLMProvider:
    from omegaprompt.providers.anthropic_provider import AnthropicProvider

    kwargs.pop("base_url", None)
    return AnthropicProvider(**kwargs)


def _build_openai(**kwargs: Any) -> LLMProvider:
    from omegaprompt.providers.openai_provider import OpenAIProvider

    return OpenAIProvider(**kwargs)


_REGISTRY = {
    "anthropic": _build_anthropic,
    "openai": _build_openai,
}
