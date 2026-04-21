# Architecture

## Design Goal

`omegacal` is a provider-neutral prompt calibration engine. The core decides how to search, score, validate, and ship. Adapters decide how to talk to a concrete provider.

That separation is the main structural defense in this refactor.

## Package Layout

- `core/`: fitness, sensitivity ranking, walk-forward, profile policy, structural risk assessment
- `domain/`: provider-neutral contracts, meta-axis types, artifacts, profile/risk schemas
- `providers/`: Anthropic, OpenAI, Gemini placeholder, local OpenAI-compatible adapters
- `targets/`: `CalibrableTarget` protocol and `PromptTarget`
- `judges/`: `RuleJudge`, `LLMJudge`, `EnsembleJudge`
- `reporting/`: artifact rendering
- `docs/`: architecture, capabilities, migration, profile/risk guidance

## Stable Core Invariants

These stayed non-negotiable:
- stress/sensitivity-based axis ranking
- top-K unlock search behavior
- `hard_gate × soft_score` fitness
- walk-forward replay on held-out data
- CI-friendly machine-readable artifact output

If an adapter cannot honor a requested feature, the core does not change its meaning. The adapter either:
- fulfills the request natively
- fulfills it with an explicit degraded capability record
- blocks it in `guarded` mode

## Core Contracts

- `ProviderRequest`
- `ProviderResponse`
- `JudgeResult`
- `EvalItemResult`
- `EvalResult`
- `CalibrationArtifact`
- `CalibrableTarget`

These contracts are semantic, not vendor-named. The search space is expressed in meta-axes:
- `system_prompt_variant`
- `few_shot_count`
- `reasoning_profile`
- `output_budget_bucket`
- `response_schema_mode`
- `tool_policy_variant`

## Execution Flow

1. Build a `PromptTarget` from provider, judge, dataset, rubric, variants, and profile.
2. Evaluate the neutral baseline.
3. Use `omega-lock` to run sensitivity ranking and top-K unlock search.
4. Replay the chosen best candidate on train and held-out test.
5. Compute walk-forward gate, boundary warnings, degraded capabilities, and ship recommendation.
6. Emit a `CalibrationArtifact`.

## Profiles

One engine, two policies:
- `guarded`: no hidden schema degradation, no non-ship-grade judge path, stricter default walk-forward thresholds
- `expedition`: allows controlled fallbacks and experimental adapters, but records each relaxation and the extra uplift it bought

Profiles do not fork the architecture. They only tune thresholds, allowed fallbacks, and recommendation policy.

## Structural Fatigue Boundaries

The engine explicitly treats these as risk boundaries:
- provider-specific assumptions leaking into the core
- weak judge paths being mistaken for ship-grade validation
- provider-native control names leaking into the search surface
- weakened walk-forward discipline
- silent capability fallback

The runtime system catches the last three directly. The first two are additionally defended by code structure and documentation.
