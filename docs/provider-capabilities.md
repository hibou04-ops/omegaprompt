# Provider Capabilities

This file is intentionally explicit about what is first-class, experimental, and placeholder.
It is part of the trust model: provider claims must match adapter code and
contract tests, not marketing language. See [trust-model.md](trust-model.md)
for how capability records affect artifact trust.

## Summary

| Provider | Tier | Status | Judge suitability | Notes |
| --- | --- | --- | --- | --- |
| Anthropic | Tier 2 cloud-grade | First-class | Ship-grade | Native strict schema, reasoning control, usage accounting |
| OpenAI | Tier 2 cloud-grade | First-class | Ship-grade | Native JSON mode, native parse path when supported |
| Gemini | Tier 2 cloud-grade | Implemented target adapter | Not ship-grade judge | Freeform, JSON object, and strict-schema paths implemented; judge remains non-ship-grade until validated |
| `local` / `ollama` / `vllm` / `llama_cpp` | Tier 3 local | Experimental target path | Exploration-grade judge only | Honest fallbacks, no fake parity |

## Anthropic

Supported:
- strict schema via native parse path
- reasoning profile translation
- JSON-oriented output
- usage accounting

Recommended use:
- target
- primary LLM judge

## OpenAI

Supported:
- strict schema when model and endpoint support native parse
- JSON object mode
- reasoning profile translation
- usage accounting

Behavior note:
- if an OpenAI-compatible endpoint rejects `reasoning_effort` or native strict parsing, the adapter reports the degradation instead of hiding it

Recommended use:
- target
- primary LLM judge

## Gemini

Current state:
- implemented provider adapter in the registry
- freeform and JSON object calls use the Google GenAI `generate_content` API
- strict-schema requests use Gemini `response_schema` when enabled and still validate locally
- `placeholder=False` in code
- `ship_grade_judge=False` until guarded-mode production probes validate Gemini judge reliability

Recommended use:
- target provider with a ship-grade cross-vendor judge for guarded runs
- expedition-only judge experiments when the artifact records the relaxed boundary

## Local Adapters

Implemented via a dedicated local OpenAI-compatible adapter family:
- `local`
- `ollama`
- `vllm`
- `llama_cpp`

Supported in this pass:
- target generation
- JSON-oriented output where the backend cooperates
- prompt-level reasoning profile hints
- explicit capability notes and degradation reporting

Not claimed in this pass:
- ship-grade strict schema
- cloud-equivalent judge semantics
- parity with frontier cloud models

Guarded behavior:
- blocks unsupported strict-schema judge paths
- blocks hidden feature degradation

Expedition behavior:
- can degrade strict schema to JSON output plus local validation
- records the degradation in the artifact
- downgrades recommendation to experimental when appropriate

## Judge Guidance

Recommended default:
- weak or local target model
- ship-grade cloud judge

Safe combinations:
- local target + Anthropic judge
- local target + OpenAI judge
- Anthropic/OpenAI target + Anthropic/OpenAI judge

Unsafe to market as ship-grade:
- local primary LLM judge without explicit expedition boundary crossing
- Gemini as primary LLM judge without explicit expedition boundary crossing and separate validation

## Claim Boundaries

Capability support means the adapter has an implemented code path and contract
tests for that path. It does not mean a provider or model is globally better
than another provider or model.

Live provider evidence belongs to the local artifact that produced it. Default
CI uses mocks and deterministic reference artifacts so provider availability,
network state, billing, and model drift do not decide whether the package is
healthy.

If a provider rejects or weakens a requested capability at runtime, the adapter
must emit a `CapabilityEvent`. Hidden degradation is a trust failure; visible
degradation is audit data.
