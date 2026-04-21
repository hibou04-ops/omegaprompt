"""Provider adapter layer for omegacal.

The kernel talks to the outside world only through this package. Each
adapter implements :class:`omegaprompt.providers.base.LLMProvider` using
its vendor's strongest native path.

v1.0 change: the two-method interface (``complete`` + ``structured_complete``)
is replaced by a single ``call(request: ProviderRequest) -> ProviderResponse``.
Meta-axes (reasoning_profile, output_budget_bucket, response_schema_mode,
tool_policy_variant) travel in the request; each adapter translates them to
vendor-native parameters internally.

Supported providers:

- ``anthropic`` - Anthropic SDK. ``messages.create`` /
  ``messages.parse(output_format=...)``, explicit prompt caching,
  adaptive thinking + effort tiers.
- ``openai`` - OpenAI SDK with ``base_url``. ``chat.completions.create`` /
  ``beta.chat.completions.parse(response_format=...)``. Works with any
  OpenAI-compatible endpoint (Azure, Groq, Together.ai, OpenRouter,
  local Ollama).
- ``gemini`` - explicit placeholder in this refactor.
- ``local`` / ``ollama`` / ``vllm`` / ``llama_cpp`` - explicit local
  OpenAI-compatible adapters with honest capability reporting.
"""

from __future__ import annotations

from omegaprompt.providers.base import (
    CapabilityEvent,
    CapabilityTier,
    LLMProvider,
    ProviderCapabilities,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    empty_usage,
    estimate_cost_units,
    max_tokens_for,
    normalize_usage,
    provider_capabilities,
    quality_per_cost,
    quality_per_latency,
    reasoning_effort_label,
    reasoning_enabled,
)
from omegaprompt.providers.factory import DEFAULT_MODELS, make_provider, supported_providers

__all__ = [
    "LLMProvider",
    "CapabilityEvent",
    "CapabilityTier",
    "ProviderCapabilities",
    "ProviderError",
    "ProviderRequest",
    "ProviderResponse",
    "empty_usage",
    "estimate_cost_units",
    "max_tokens_for",
    "normalize_usage",
    "provider_capabilities",
    "quality_per_cost",
    "quality_per_latency",
    "reasoning_effort_label",
    "reasoning_enabled",
    "DEFAULT_MODELS",
    "make_provider",
    "supported_providers",
]
