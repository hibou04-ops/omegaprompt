# Provider Capabilities

This file is intentionally explicit about what is first-class, experimental, and placeholder.

## Summary

| Provider | Tier | Status | Judge suitability | Notes |
| --- | --- | --- | --- | --- |
| Anthropic | Tier 2 cloud-grade | First-class | Ship-grade | Native strict schema, reasoning control, usage accounting |
| OpenAI | Tier 2 cloud-grade | First-class | Ship-grade | Native JSON mode, native parse path when supported |
| Gemini | Tier 2 cloud-grade | Placeholder | Not ship-grade | Adapter reserved, not implemented in this pass |
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
- explicit placeholder only
- present in the registry so the core stays provider-neutral
- not implemented deeply enough to claim parity

Recommended use:
- none for production in this pass

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
- strong cloud judge

Safe combinations:
- local target + Anthropic judge
- local target + OpenAI judge
- Anthropic/OpenAI target + Anthropic/OpenAI judge

Unsafe to market as ship-grade:
- local primary LLM judge without explicit expedition boundary crossing
- Gemini placeholder in any judge position
