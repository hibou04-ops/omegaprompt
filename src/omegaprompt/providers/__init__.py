"""Provider adapter layer.

The kernel talks to the outside world only through this package. Each
adapter implements :class:`omegaprompt.providers.base.LLMProvider` using
its vendor's strongest native path.

v1.0 change: the two-method interface (``complete`` + ``structured_complete``)
is replaced by a single ``call(request: ProviderRequest) -> ProviderResponse``.
Meta-axes (reasoning_profile, output_budget, response_schema_mode,
tool_policy) travel in the request; each adapter translates them to
vendor-native parameters internally.

Supported providers:

- ``anthropic`` - Anthropic SDK. ``messages.create`` /
  ``messages.parse(output_format=...)``, explicit prompt caching,
  adaptive thinking + effort tiers.
- ``openai`` - OpenAI SDK with ``base_url``. ``chat.completions.create`` /
  ``beta.chat.completions.parse(response_format=...)``. Works with any
  OpenAI-compatible endpoint (Azure, Groq, Together.ai, OpenRouter,
  local Ollama).
"""

from __future__ import annotations

from omegaprompt.providers.base import (
    LLMProvider,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    empty_usage,
    max_tokens_for,
    normalize_usage,
    reasoning_effort_label,
    reasoning_enabled,
)
from omegaprompt.providers.factory import DEFAULT_MODELS, make_provider, supported_providers

__all__ = [
    "LLMProvider",
    "ProviderError",
    "ProviderRequest",
    "ProviderResponse",
    "empty_usage",
    "max_tokens_for",
    "normalize_usage",
    "reasoning_effort_label",
    "reasoning_enabled",
    "DEFAULT_MODELS",
    "make_provider",
    "supported_providers",
]
