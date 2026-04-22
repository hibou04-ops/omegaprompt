# omegaprompt

> **New to this?** Start here: [EASY_README.md](EASY_README.md) (English) · [EASY_README_KR.md](EASY_README_KR.md) (한국어). Compressed plain-language introductions for readers who find the full doc below intimidating.

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)
[![PyPI](https://img.shields.io/badge/pypi-1.1.0-blue.svg)](https://pypi.org/project/omegaprompt/)
[![Tests](https://img.shields.io/badge/tests-149%20passing-brightgreen.svg)](tests/)
[![Providers](https://img.shields.io/badge/providers-anthropic%20%7C%20openai%20%7C%20openai--compatible%20%7C%20local-informational.svg)](#7-provider-adapters)
[![Schema](https://img.shields.io/badge/artifact-schema%20v2.0-blueviolet.svg)](#8-the-calibrationartifact-schema-v20)
[![Parent framework](https://img.shields.io/badge/framework-omega--lock-blueviolet.svg)](https://github.com/hibou04-ops/omega-lock)

> **Abstract.** `omegaprompt` is a provider-neutral calibration engine for LLM prompts. It takes the machine-learning defenses against overfitting — train/test split with a pre-declared gate, sensitivity-driven axis unlock, hard-gate × soft-score fitness — and applies them to prompt engineering without coupling to any single vendor's API surface. The public contract is expressed as semantic *meta-axes* (reasoning profile, output budget, response schema mode, tool policy) that each adapter translates to its vendor's native parameters. Adapters declare their capabilities up front; runtime degradations are recorded as `CapabilityEvent`s in the calibration artifact. Two execution profiles (`guarded` / `expedition`) make the trade between validation strength and exploratory reach explicit. The `CalibrationArtifact` (schema v2.0) records the neutral-baseline and calibrated runs side by side, so a reviewer can see not only *what* shipped but *how much* calibration earned over doing nothing.

```bash
pip install omegaprompt
```

The main package is self-contained. Two optional sub-tools (`mini-omega-lock`, `mini-antemortem-cli`) distribute **separately** and plug in via the `omegaprompt.preflight` interface to add empirical and analytical preflight measurements respectively. Standalone users do not need them — see §5.8. Korean README: [README_KR.md](README_KR.md).

---

## Table of contents

- [1. Problem statement](#1-problem-statement)
- [2. Contributions](#2-contributions)
- [3. System architecture](#3-system-architecture)
- [4. Key abstractions](#4-key-abstractions)
- [5. The calibration pipeline](#5-the-calibration-pipeline)
- [6. The three judges](#6-the-three-judges)
- [7. Provider adapters](#7-provider-adapters)
- [8. The CalibrationArtifact (schema v2.0)](#8-the-calibrationartifact-schema-v20)
- [9. CLI surface](#9-cli-surface)
- [10. Quick start](#10-quick-start)
- [11. Worked example](#11-worked-example)
- [12. Validation](#12-validation)
- [13. Comparative positioning](#13-comparative-positioning)
- [14. Limitations and scope boundaries](#14-limitations-and-scope-boundaries)
- [15. Roadmap](#15-roadmap)
- [16. Prior art and credits](#16-prior-art-and-credits)
- [Appendix A: data contracts](#appendix-a-data-contracts)
- [Appendix B: meta-axis to vendor-parameter mapping](#appendix-b-meta-axis-to-vendor-parameter-mapping)
- [Appendix C: invariants](#appendix-c-invariants)
- [Appendix D: AdaptationPlan contract](#appendix-d-adaptationplan-contract)
- [Citing](#citing)
- [License](#license)

---

## 1. Problem statement

Prompt engineering in production settings has four failure modes that no amount of manual iteration resolves, because each one is structural rather than a matter of skill.

### 1.1 Overfitting to the training set

A practitioner curates a small set of example inputs, iterates prompt variants against them, picks the top scorer, and ships. On day two of production, the prompt encounters inputs the training set did not represent, and the quality gap between *"scored 4.8 on the examples I picked"* and *"scores 4.8 on inputs I will receive"* opens up. This is the textbook definition of overfitting. The defense has been known since the 1990s: a held-out test slice that the optimiser never sees, evaluated only at ship time, under a correlation threshold declared before the scores are computed. Every ML curriculum teaches this. Nearly every prompt-engineering workflow skips it.

### 1.2 Self-agreement bias in LLM-as-judge

When the target model and the grading model come from the same vendor — or worse, are the same model — the judge's biases overlap with the target's biases. A response that flatters the vendor's training distribution can pass grading that a disinterested second vendor would fail. The judge is then not an independent assessor but a peer. Standard defences (stronger judge, different model, different vendor) require the pipeline to treat the two call sites as *independent* at the API boundary, not as "pick one API key and reuse it."

### 1.3 Vendor-coupling in calibration axes

Most prompt-optimisation tools inherit their search axes from the most convenient vendor's API surface: `temperature`, `top_p`, `max_tokens`, `effort`, `thinking_enabled`. These names are artefacts of a specific API contract. Calibrating with them couples the calibration *discipline* to a particular vendor's ergonomics, and the artifact becomes unreadable when the target model migrates. A search that discovered *"effort = high improves this task"* is also a search that cannot be replayed on a model with no effort parameter.

### 1.4 Hidden fallbacks and silent degradation

Real LLM SDKs degrade silently. A reasoning-effort parameter is rejected by a local endpoint, but the request proceeds without it. A structured-output path is unavailable on a given model, so JSON gets regex-parsed. A cache-control header is ignored on some providers, inflating token cost. In each case the calibration still produces numbers. The numbers are no longer comparable across providers, and the operator has no record of what changed. A calibration framework that does not name these degradations makes them invisible to the CI pipeline downstream.

`omegaprompt` is the response to all four. Each contribution below targets one or more of these failure modes.

---

## 2. Contributions

1. **Provider-neutral meta-axes.** The public search space is expressed in semantic categories (reasoning profile, output budget bucket, response schema mode, tool policy variant) rather than vendor-specific parameter names. Each provider adapter maps meta-axes to its vendor's native surface internally. The calibration artifact records the meta-axis value (e.g. `reasoning_profile: deep`), not the translated parameter (e.g. `effort: high`), so the same artifact is legible and replayable across vendors.

2. **Execution profiles.** A `guarded` profile (default) refuses to silently relax validation — unship-grade judges raise, structured-schema fallback to prose raises, hidden capability loss raises. An `expedition` profile permits controlled boundary crossing, but every relaxation is recorded as a `RelaxedSafeguard` entry on the artifact. The two profiles make the bargain between strictness and reach explicit and auditable.

3. **Capability tiers and explicit degradation events.** Each provider declares a `ProviderCapabilities` record (supports strict schema, json object, reasoning profiles, usage accounting, LLM judging, tools; tier CORE / CLOUD / LOCAL; experimental / placeholder flags). When an adapter degrades at runtime — for instance, retries without a rejected `reasoning_effort` parameter — it emits a `CapabilityEvent` capturing the capability, the requested value, the applied fallback, and a user-visible note. The event flows up through `EvalItemResult` → `EvalResult` → `CalibrationArtifact` so downstream diffs can detect capability regressions.

4. **Neutral-baseline vs calibrated comparison.** The `CalibrationArtifact` (schema v2.0) records the fitness of the neutral-parameter baseline and the fitness of the calibrated best side by side, with absolute and percent uplift, plus quality-per-cost and quality-per-latency ratios at both points. A reviewer sees not just *the best score* but *what the search earned over doing nothing*.

5. **Walk-forward ship gate with pre-declared thresholds.** The held-out test evaluation uses a Pearson-correlation threshold (`--min-kc4`) and a generalisation-gap threshold (`--max-gap`) that default from the execution profile and are recorded on the artifact. The thresholds cannot be lowered after the scores are seen; this is the Winchester defence, borrowed from quant finance. `status = FAIL_KC4_GATE` is a ship-blocker by construction.

6. **Judge protocol with three shipped implementations.** `LLMJudge` uses a provider's strict-schema parse path; `RuleJudge` runs deterministic Python predicates for format / refusal / regex gates at zero API cost; `EnsembleJudge` short-circuits LLM grading when rule gates fail. The three compose under a single `Judge` protocol, which the `PromptTarget` consumes without knowledge of which strategy is wired in.

Together, these contributions turn prompt calibration from an ergonomic exercise into an auditable engineering pipeline whose output is a CI-gate-ready artifact rather than a spreadsheet of scores.

---

## 3. System architecture

### 3.1 Layered package structure

```
omegaprompt/
├── domain/        Provider-neutral contracts (enums, dataset, rubric,
│                  params, result, profiles). Depends on nothing.
├── core/          Calibration kernel (fitness, artifact I/O, walk-forward,
│                  sensitivity ranking, profile policy, run risk). Depends
│                  only on domain.
├── providers/     LLMProvider Protocol + adapter implementations
│                  (Anthropic, OpenAI / OpenAI-compatible, Gemini placeholder,
│                  local/ollama/vllm/llama_cpp). Translates meta-axes to
│                  vendor parameters, reports capabilities and degradation.
├── judges/        Judge protocol + LLM / Rule / Ensemble implementations.
│                  Depends on domain and providers.
├── targets/       CalibrableTarget protocol + PromptTarget adapter. The
│                  composition point where omega-lock's search layer plugs in.
├── reporting/     Artifact → Markdown renderer.
├── commands/      Typer subcommands: calibrate, report, diff.
└── cli.py         Top-level Typer application.
```

### 3.2 Dependency direction

```
       ┌──────────────────────────────────────────────────────────┐
       │                     domain/                              │
       │ (enums, dataset, params, rubric, result, profiles)       │
       └──────────────────┬───────────────────────────────────────┘
                          │ imported by
                          ▼
       ┌──────────────────────────────────────────────────────────┐
       │                     core/                                │
       │ (fitness, artifact, walkforward, sensitivity, policy)    │
       └──────────────────┬───────────────────────────────────────┘
                          │
                          ▼
       ┌──────────────────────────────────────────────────────────┐
       │                     providers/                           │
       │ (LLMProvider, ProviderRequest/Response, capabilities,    │
       │  factory, Anthropic/OpenAI/local adapters)               │
       └──────────────────┬───────────────────────────────────────┘
                          │
                          ▼
       ┌──────────────────────────────────────────────────────────┐
       │                     judges/                              │
       │ (Judge protocol; LLMJudge, RuleJudge, EnsembleJudge)     │
       └──────────────────┬───────────────────────────────────────┘
                          │
                          ▼
       ┌──────────────────────────────────────────────────────────┐
       │                     targets/                             │
       │ (CalibrableTarget, PromptTarget)                         │
       └──────────────────┬───────────────────────────────────────┘
                          │
                          ▼
       ┌──────────────────────────────────────────────────────────┐
       │              omega-lock search engine                    │
       │   (stress, top-K unlock, grid search, walk-forward)      │
       └──────────────────┬───────────────────────────────────────┘
                          │ produces
                          ▼
       ┌──────────────────────────────────────────────────────────┐
       │   CalibrationArtifact (schema_version=2.0, JSON on disk) │
       └──────────────────────────────────────────────────────────┘
```

The dependency graph has no back-edges. `domain` does not import from anywhere inside `omegaprompt`. `core` knows about `domain` only. `providers` and `judges` never import search or target code. `targets` is the single composition point where the adapter layer plugs into `omega-lock`'s search engine via the `CalibrableTarget` protocol.

### 3.3 Boundary between discipline and adapters

The calibration *discipline* — sensitivity measurement, top-K unlock, grid search, walk-forward with pre-declared gates, hard-gate × soft-score fitness, artifact schema — is vendor-agnostic and lives in `core/` and `domain/`. The *adapter layer* — how a `reasoning_profile: deep` becomes a vendor-native API call, how a vendor's usage record is normalised to `input_tokens / output_tokens / cache_creation_input_tokens / cache_read_input_tokens` — lives in `providers/`. A reader evaluating the integrity of the calibration can review `core/` and `domain/` without caring which vendors are wired in. A reader onboarding a new provider can implement `LLMProvider` without reading the search layer.

---

## 4. Key abstractions

### 4.1 Meta-axes

Six axes constitute the public search space:

| Axis | Type | Semantics | Vendor-native translation example |
|---|---|---|---|
| `system_prompt_variant` | `int` | Index into `PromptVariants.system_prompts`. | Message-level system-prompt substitution. |
| `few_shot_count` | `int` | Count of examples from `PromptVariants.few_shot_examples`. | Message-list prefix length. |
| `reasoning_profile` | enum `OFF / LIGHT / STANDARD / DEEP` | How much reasoning effort the target should spend. | Anthropic `thinking={"type":"adaptive"}` + `effort` in `{low,medium,high}`; OpenAI `reasoning_effort`; local: system-prompt suffix. |
| `output_budget_bucket` | enum `SMALL / MEDIUM / LARGE` | Discretised `max_tokens`. | Resolved to `1024 / 4096 / 16000`. |
| `response_schema_mode` | enum `FREEFORM / JSON_OBJECT / STRICT_SCHEMA` | How strictly the response is shape-constrained. | Anthropic `messages.create` vs `messages.parse(output_format=...)`; OpenAI `chat.completions.create` vs `beta.chat.completions.parse(response_format=...)` vs `response_format={"type":"json_object"}`. |
| `tool_policy_variant` | enum `NO_TOOLS / TOOL_OPTIONAL / TOOL_REQUIRED` | Tool-use policy. | No-op for plain chat targets; `tool_choice` derivative on tool-capable targets. |

The `MetaAxisSpace` record declares which values of each enum are in scope for a particular run; a single-member list locks the axis at a fixed value. The `ResolvedPromptParams` record carries the concrete choices after the searcher picks. Both records are Pydantic models with `extra="forbid"` — unknown keys raise at parse time.

### 4.2 Execution profiles

Two profiles capture the practitioner's position on the strict-versus-exploratory trade-off.

```python
class ExecutionProfile(str, Enum):
    GUARDED   = "guarded"     # default; blocks hidden fallbacks
    EXPEDITION = "expedition" # permits recorded boundary crossing
```

Guarded mode:
- Refuses to use a provider whose `supports_llm_judge` capability is false as a judge.
- Raises when a `STRICT_SCHEMA` request hits a provider that cannot honour it.
- Treats `experimental` or `placeholder` adapters as ineligible for ship-grade positions.
- Uses strict defaults for `max_gap` and `min_kc4` on the walk-forward gate.

Expedition mode:
- Permits the above, but every relaxation is recorded as a `RelaxedSafeguard` on the artifact, and `stayed_within_guarded_boundaries` is set to `False`.
- `additional_uplift_from_boundary_crossing` records how much of the calibrated fitness came from work that guarded mode would have blocked, so the reviewer can see whether the boundary crossing actually paid off.

Profile selection is a single CLI flag (`--profile guarded|expedition`) and appears on the artifact as `selected_profile`.

### 4.3 Provider capability model

Every `LLMProvider` exposes a `capabilities() -> ProviderCapabilities` method:

```python
class ProviderCapabilities(BaseModel):
    provider: str
    tier: CapabilityTier                    # CORE / CLOUD / LOCAL
    supports_strict_schema: bool = False
    supports_json_object: bool = False
    supports_reasoning_profiles: bool = False
    supports_usage_accounting: bool = True
    supports_llm_judge: bool = False
    ship_grade_judge: bool = False
    supports_tools: bool = False
    experimental: bool = False
    placeholder: bool = False
    notes: list[str]
```

Capability *tiers* are a coarse classification:

| Tier | Purpose | Example |
|---|---|---|
| `tier_1_core_parity` | Neutral contracts and calibration kernel. Required. | In-memory test stubs, legacy provider shims. |
| `tier_2_cloud_grade` | First-class judge-eligible cloud providers. | Anthropic, OpenAI (frontier models). |
| `tier_3_local` | Local OpenAI-compatible backends. Target-eligible; by default not ship-grade judges. | Ollama, vLLM, llama.cpp, local OpenAI-compatible servers. |

Tiers are a policy input: the guarded profile refuses tier-3 providers in the judge position. Expedition permits it, recording a `RelaxedSafeguard`.

### 4.4 Capability events

When an adapter degrades at runtime, it emits a structured record:

```python
class CapabilityEvent(BaseModel):
    capability: str          # e.g. "reasoning_profile"
    requested: str           # "deep"
    applied: str             # "off"
    reason: str              # "endpoint rejected reasoning_effort"
    user_visible_note: str   # actionable English explanation
    affects_guarded_boundary: bool = True
```

Events propagate from the `ProviderResponse` up through the `EvalItemResult` into the `EvalResult.degraded_capabilities`, and finally onto `CalibrationArtifact.degraded_capabilities`. A reader of the artifact can grep for capability names to see which features were not honoured during the run. In guarded mode, events with `affects_guarded_boundary=True` block the run; in expedition mode they merely record.

### 4.5 Ship recommendations

The artifact's `ship_recommendation` field takes one of:

```python
class ShipRecommendation(str, Enum):
    SHIP = "ship"
    HOLD = "hold"
    ROLLBACK = "rollback"
```

Computation is deterministic from `status`, walk-forward outcome, hard-gate pass rate, `stayed_within_guarded_boundaries`, and the presence of blocking `CapabilityEvent`s. Same artifact in, same recommendation out — a CI pipeline whitelists `SHIP` without interpreting prose.

---

## 5. The calibration pipeline

### 5.1 Inputs

Three files, all user-authored, all Pydantic-validated:

- **`dataset.jsonl`** — one `DatasetItem` per line: `id`, `input`, optional `reference`, optional `metadata`.
- **`rubric.json`** — `JudgeRubric` with per-dimension weight and integer scale, plus hard gates each labelled with an `evaluator` (`rule` / `judge` / `post`).
- **`variants.json`** — `PromptVariants` with `system_prompts` pool and optional `few_shot_examples`.

Optionally: `space.json` (custom `MetaAxisSpace`), `test.jsonl` (held-out slice for walk-forward).

### 5.2 Sensitivity measurement

Around a neutral-parameter baseline, the searcher perturbs each meta-axis across its declared values and records the fitness delta. Axes are ranked by the Gini coefficient of their fitness-delta distribution — high Gini = concentrated, high-leverage; low Gini = diffuse, low-signal. Sensitivity is the *a priori* case for spending search budget on an axis.

### 5.3 Top-K unlock

The top `--unlock-k` axes by Gini delta enter the grid-search subspace. The rest stay locked at their neutral values. This cuts search cost from `Π(|axis|)` over all axes to `Π(|axis|)` over the top-K, typically a 5–20× reduction for `k=3`.

### 5.4 Grid search

Every combination in the unlocked subspace is evaluated. Each evaluation issues one provider call per dataset item (target) plus one judge call per item (if the judge is `LLMJudge` or `EnsembleJudge` with the LLM fallback triggered). The returned `EvalResult` records fitness, per-item scores, aggregate token usage, latency, and any capability events.

### 5.5 Walk-forward replay

The training-best parameters are replayed on the held-out test slice. The replay uses the *same* `PromptTarget` adapter with a different dataset; there is no leakage possible because the test slice was never seen by the searcher.

### 5.6 KC-4 Pearson gate

The Pearson correlation between train per-item scores and test per-item scores (on the shared dataset ids) is compared to `--min-kc4`. The generalisation gap `|train - test| / |train|` is compared to `--max-gap`. A failure on either sets `status = FAIL_KC4_GATE` and `ship_recommendation = HOLD`. Both thresholds are recorded on the artifact; they cannot be lowered after the fact.

### 5.7 Artifact emission

The `CalibrationArtifact` (see §8) is written as pretty-printed JSON to the `--output` path. It carries enough information to (a) render a Markdown report, (b) diff against a prior run, (c) gate CI on machine-readable fields without parsing prose.

### 5.8 Preflight and adaptation (optional sub-tool ecosystem)

The main pipeline does not assume that its default thresholds (`min_kc4 = 0.5`, `max_gap = 0.25`, `unlock_k = 3`) are universally correct. `omegaprompt.preflight` defines a stable plugin contract for two *optional* external sub-tools that measure the actual environment and emit a shared :class:`AdaptationPlan` the main pipeline consumes. The discipline's *defenses* — hard-gate fitness collapse, walk-forward ship gate, sensitivity-driven axis unlock — remain in place; only the numeric parameters are tuned to the environment.

**Standalone `omegaprompt` ships no preflight probe code.** Most users never need it. The preflight module exposes only:

- **Contracts** — Pydantic types (`PreflightReport`, `AnalyticalFinding`, `JudgeQualityMeasurement`, `EndpointMeasurement`, `PerformanceMeasurement`) that external sub-tools emit.
- **Adaptation logic** — `derive_adaptation_plan(report)` maps a report to an `AdaptationPlan`; `apply_adaptation_plan(plan, ...)` clips the plan against the caller's defaults so adaptation can only *strengthen* the discipline.

Two external sub-tools plug in:

| Sub-tool | Repository / PyPI | Role |
|---|---|---|
| **`mini-omega-lock`** | `pip install mini-omega-lock` (separate) | **Empirical preflight.** Probes the live judge + endpoint to measure consistency, schema reliability, context margin, latency, noise floor. Emits `JudgeQualityMeasurement`, `EndpointMeasurement`, `PerformanceMeasurement`. |
| **`mini-antemortem-cli`** | `pip install mini-antemortem-cli` (separate) | **Analytical preflight.** Reads the run configuration and classifies calibration trap patterns (self-agreement bias, small-sample KC-4 power, rubric concentration, variant homogeneity, …) as `REAL` / `GHOST` / `NEW` / `UNRESOLVED`. Emits `AnalyticalFinding` records. |

Either can be used alone; both compose into the same `PreflightReport`.

When a sub-tool runs and feeds the result into `derive_adaptation_plan`, the derivation rules *only strengthen* the discipline:

```
noise_floor >= 0.05  → min_kc4: max(default, 0.50)
noise_floor >= 0.15  → min_kc4: max(default, 0.60)
noise_floor >= 0.25  → min_kc4: max(default, 0.70)
noise_floor >= 0.35  → min_kc4: max(default, 0.80)

judge_consistency < 0.60  → rescore_count = 3 (median)
judge_consistency < 0.80  → rescore_count = 2
judge_consistency < 0.70  → judge_ensemble_shift = 0.40 (RuleJudge weight up)

schema_reliability < 0.90  → schema_mode_fallback = JSON_OBJECT

projected_wall_time > 4h and unlock_k > 1  → unlock_k -= 1

small_sample_kc4_power finding (HIGH) → max_gap: min(0.40, default * 1.6)
variants_homogeneous (REAL/NEW)       → skip_axes += ["system_prompt_variant"]
```

`apply_adaptation_plan(plan, min_kc4=..., max_gap=..., unlock_k=...)` uses `max` on `min_kc4`, `min` on `max_gap`, and `min` on `unlock_k`, so a plan that attempts to widen tolerance is clipped to the caller's configuration (see Appendix C, invariant 10 and the accompanying test `test_apply_plan_never_weakens_kc4`).

Standalone `omegaprompt` ignores the whole subsystem and runs with its declared defaults. A calibration augmented with `mini-omega-lock` + `mini-antemortem-cli` produces a plan whose overrides are fully auditable on the artifact, and the pipeline adapts within the discipline rather than failing loud on weak infrastructure.

---

## 6. The three judges

```python
class Judge(Protocol):
    name: str

    def score(
        self,
        *,
        rubric: JudgeRubric,
        item: DatasetItem,
        target_response: str,
    ) -> tuple[JudgeResult, dict[str, int]]: ...
```

### 6.1 LLMJudge

Delegates to an `LLMProvider` via `ResponseSchemaMode.STRICT_SCHEMA`. The vendor's native parse path (`messages.parse` on Anthropic, `beta.chat.completions.parse` on OpenAI) returns a Pydantic-validated `JudgeResult` — no regex fallback, no prose-to-structure inference. Under guarded mode, `LLMJudge` refuses to run on a provider whose `supports_llm_judge` capability is false.

### 6.2 RuleJudge

Evaluates only hard gates that declare `evaluator="rule"` in the rubric. Each rule is a deterministic Python callable (`default_no_refusal()`, `default_non_empty()`, `json_object_check()`, `regex_check(name, pattern)`, or a user-supplied lambda). No LLM calls, zero API cost, reproducible across runs. Dimensions go unscored; `RuleJudge` is typically composed with `LLMJudge` rather than used alone.

### 6.3 EnsembleJudge

Runs `RuleJudge` first. If any rule gate fails, the result short-circuits with the rule gate results and no LLM call. If every rule gate passes, escalates to the fallback judge (typically `LLMJudge`) for dimension scoring and judge-gate evaluation. The two judges' gate results are merged on return. In practice, `EnsembleJudge` recovers ~0.5–0.9× the LLM-judge cost depending on how often responses fail structural gates.

```python
from omegaprompt import EnsembleJudge, LLMJudge, RuleJudge, make_provider
from omegaprompt.judges.rule_judge import default_no_refusal, json_object_check

judge_provider = make_provider("anthropic")
rule = RuleJudge(checks=[default_no_refusal(), json_object_check("format_valid")])
llm = LLMJudge(provider=judge_provider)
judge = EnsembleJudge(rule_judge=rule, fallback=llm)
```

---

## 7. Provider adapters

### 7.1 Capability declaration

Every adapter declares a `ProviderCapabilities` record. Built-in adapters:

| Provider | Tier | Strict schema | JSON object | Reasoning | Ship-grade judge | Notes |
|---|---|---|---|---|---|---|
| `anthropic` | cloud | yes | yes | yes | yes | `messages.parse` + explicit `cache_control`. |
| `openai` | cloud | yes | yes | yes | yes | `beta.chat.completions.parse`; drops `reasoning_effort` on unsupported endpoints and records the event. |
| `gemini` | cloud | no | no | no | no | **Placeholder** — raises a clear error; slot reserved for the forthcoming native adapter. |
| `ollama` / `local` / `vllm` / `llama_cpp` | local | best-effort | yes | no | no | Target-eligible; refuses LLM-judge position under guarded mode. |

### 7.2 Anthropic

`messages.create` for freeform and JSON-object modes (with a system-prompt suffix instructing JSON output, since Anthropic does not expose a native `response_format={"type":"json_object"}`); `messages.parse(output_format=T)` for `STRICT_SCHEMA`. The system block is always wrapped with `cache_control={"type":"ephemeral"}` so repeated judge calls in a calibration run hit the prompt cache. Reasoning profiles map to `thinking={"type":"adaptive"}` plus `output_config.effort`.

### 7.3 OpenAI and OpenAI-compatible

`chat.completions.create` for freeform and `response_format={"type":"json_object"}` for JSON mode; `beta.chat.completions.parse(response_format=T)` for `STRICT_SCHEMA`. `reasoning_effort` is attempted for non-OFF reasoning profiles; when the endpoint rejects the parameter (some compatible endpoints do), the adapter retries without it and emits a `CapabilityEvent` naming the fallback. Accepts a `base_url`, which makes every OpenAI-compatible endpoint (Azure OpenAI, Groq, Together.ai, OpenRouter, local vLLM / Ollama) a drop-in target or judge.

### 7.4 Gemini (placeholder)

Reserved in the provider registry so that the migration from placeholder to native adapter is contract-stable: downstream code already names `gemini` as a valid provider string; only the adapter implementation needs to fill in.

### 7.5 Local endpoints

Local OpenAI-compatible backends are first-class target providers but are not considered ship-grade judges by default. The guarded profile blocks their use in the judge position; expedition mode records the relaxation. This is a policy position, not a library limitation — a local model that demonstrates ship-grade judge quality on your domain can have its capability override set explicitly.

### 7.6 Extending

```python
class LLMProvider(Protocol):
    name: str
    model: str
    def call(self, request: ProviderRequest) -> ProviderResponse: ...
    def capabilities(self) -> ProviderCapabilities: ...
```

Implement the two methods. Register in `providers/factory.py`. Nothing else changes — the search layer, the judges, the targets, the artifact schema all remain unchanged.

---

## 8. The CalibrationArtifact (schema v2.0)

The schema is deliberately rich. The artifact is the system of record for the run, and reviewers should not have to re-derive anything.

```json
{
  "schema_version": "2.0",
  "engine_name": "omegaprompt",
  "method": "p1",
  "unlock_k": 3,
  "selected_profile": "guarded",

  "neutral_baseline_params": {
    "system_prompt_variant": 0,
    "few_shot_count": 0,
    "reasoning_profile": "standard",
    "output_budget_bucket": "medium"
  },
  "neutral_fitness": 0.612,

  "calibrated_params": {
    "system_prompt_variant": 2,
    "few_shot_count": 1,
    "reasoning_profile": "deep"
  },
  "calibrated_fitness": 0.876,

  "best_params": { "...": "mirror of calibrated_params for backward-compat" },
  "best_fitness": 0.876,

  "uplift_absolute": 0.264,
  "uplift_percent": 43.14,
  "quality_per_cost_neutral":    0.00012,
  "quality_per_cost_best":       0.00009,
  "quality_per_latency_neutral": 0.00071,
  "quality_per_latency_best":    0.00055,

  "walk_forward": {
    "train_best_fitness": 0.876,
    "test_fitness":       0.841,
    "generalization_gap": 0.040,
    "kc4_correlation":    0.84,
    "passed": true
  },

  "hard_gate_pass_rate": 1.00,
  "sensitivity_ranking": [
    { "axis": "system_prompt_variant", "gini_delta": 0.52, "rank": 0 },
    { "axis": "reasoning_profile",     "gini_delta": 0.33, "rank": 1 },
    { "axis": "few_shot_count",        "gini_delta": 0.11, "rank": 2 }
  ],

  "boundary_warnings": [],
  "degraded_capabilities": [],
  "relaxed_safeguards": [],
  "stayed_within_guarded_boundaries": true,
  "additional_uplift_from_boundary_crossing": 0.0,
  "guarded_boundary_crossed": false,

  "ship_recommendation": "ship",
  "status": "OK",
  "rationale": "passed",

  "target_provider": "openai",
  "target_model":    "gpt-4o",
  "target_capabilities": { "tier": "tier_2_cloud_grade", "supports_strict_schema": true, "ship_grade_judge": true, "...": "..." },
  "judge_provider": "anthropic",
  "judge_model":    "claude-opus-4-7",
  "judge_capabilities": { "tier": "tier_2_cloud_grade", "supports_llm_judge": true, "ship_grade_judge": true, "...": "..." },

  "usage_summary": { "input_tokens": 4820, "output_tokens": 2110, "cache_read_input_tokens": 12088 },
  "latency_summary_ms": { "target_p50": 742, "judge_p50": 1103 },
  "cost_basis": "normalized_token_units",
  "n_candidates_evaluated": 12,
  "total_api_calls": 96
}
```

The key structural choice: `neutral_baseline` and `calibrated` are recorded side by side, with explicit `uplift_absolute` / `uplift_percent` fields. A reviewer sees not just "the best score was 0.876" but "the search moved this from 0.612 to 0.876, a 43% improvement, at a 33% cost increase per unit of quality." That framing is the difference between "we ran a calibration" and "calibration was worth it on this workload."

---

## 9. CLI surface

### 9.1 `omegaprompt calibrate`

End-to-end run: parse inputs, build the target + judge, invoke `omega_lock.run_p1`, emit the `CalibrationArtifact`.

```bash
omegaprompt calibrate train.jsonl \
  --rubric rubric.json \
  --variants variants.json \
  --test test.jsonl \
  --profile guarded \
  --target-provider openai   --target-model gpt-4o \
  --judge-provider anthropic --judge-model claude-opus-4-7 \
  --method p1 --unlock-k 3 \
  --output artifact.json
```

Exit codes: `0` on success (regardless of `status`), `2` on environment problems (missing env var, unknown provider, missing `omega-lock`).

### 9.2 `omegaprompt report <artifact.json>`

Renders the artifact as Markdown (for PR descriptions, CI step outputs, human review).

```bash
omegaprompt report artifact.json > report.md
```

### 9.3 `omegaprompt diff <old.json> <new.json>`

Compares two artifacts. Exits non-zero when the new run regresses on any of: `calibrated_fitness`, `walk_forward.test_fitness`, `hard_gate_pass_rate`, `quality_per_cost_best`, `quality_per_latency_best`, or `stayed_within_guarded_boundaries` (true-to-false is a regression). Intended for CI use.

```bash
omegaprompt diff previous.json artifact.json   # exit 1 on regression
```

The `omegaprompt` CLI binary remains as a compatibility alias during migration.

---

## 10. Quick start

```bash
pip install omegaprompt
```

A minimal run (Anthropic target + Anthropic judge, guarded profile):

```bash
omegaprompt calibrate examples/sample_dataset.jsonl \
  --rubric examples/rubric_example.json \
  --variants examples/variants_example.json \
  --target-provider anthropic \
  --judge-provider anthropic \
  --profile guarded \
  --output artifact.json
```

Cross-vendor (OpenAI target, Anthropic judge) to break self-agreement:

```bash
omegaprompt calibrate train.jsonl \
  --rubric rubric.json --variants variants.json --test test.jsonl \
  --target-provider openai   --target-model gpt-4o \
  --judge-provider anthropic --judge-model claude-opus-4-7 \
  --output artifact.json
```

Local target (Ollama) + cloud judge:

```bash
omegaprompt calibrate train.jsonl \
  --rubric rubric.json --variants variants.json --test test.jsonl \
  --target-provider ollama \
    --target-base-url http://localhost:11434/v1 \
    --target-model llama3.1:8b \
  --judge-provider openai --judge-model gpt-4o \
  --profile guarded \
  --output artifact.json
```

Render and diff:

```bash
omegaprompt report artifact.json
omegaprompt diff previous.json artifact.json
```

---

## 11. Worked examples

### 11.1 The failure mode the tool catches (illustrative)

A practitioner iterates prompts against a 10-item training set and compares two candidates.

```
Candidate A:  train_fitness = 0.923   (4.6 / 5 average)
Candidate B:  train_fitness = 0.876   (4.4 / 5 average)
```

Candidate A wins on training. The practitioner ships A.

Walk-forward on a held-out test slice the searcher never saw:

```
Candidate A:  train = 0.923   test = 0.612   gap = 33.7%
Candidate B:  train = 0.876   test = 0.841   gap =  4.0%

run_p1 status: FAIL_KC4_GATE
Reason: pearson(train_per_item, test_per_item) < --min-kc4
Ship recommendation: HOLD.
Rationale: candidate A's per-item train ranking does not predict its per-item test ranking;
           the calibration signal did not generalise.
```

Candidate A overfit the training slice. `omegaprompt` blocks the ship decision mechanically, before the practitioner sees the production behaviour. Candidate B — lower training score, dramatically better generalisation — is the correct decision.

The artifact records both candidates in the grid history, the failed KC-4, and `ship_recommendation: "hold"`. CI gating on `stayed_within_guarded_boundaries == true` and `ship_recommendation == "ship"` blocks the merge without the practitioner needing to parse prose.

> Numbers in this subsection are illustrative of the failure mode. A reproducible, machine-generated example is in §11.2.

### 11.2 Reproducible reference run

The repository ships a deterministic reference run (`examples/reference/reproduce_reference_artifact.py`) that drives the **real** `omega_lock.run_p1` against an in-memory target + judge whose fitness is a closed-form function of the meta-axis parameters. No LLM API calls, no network, no randomness. The same run produces byte-identical output on every machine.

To reproduce the numbers below:

```bash
python examples/reference/reproduce_reference_artifact.py
# writes examples/reference/reference_artifact.json
```

The machine-produced artifact (trimmed; full file at `examples/reference/reference_artifact.json`):

```json
{
  "schema_version": "2.0",
  "engine_name": "omegaprompt",
  "method": "p1",
  "unlock_k": 2,
  "selected_profile": "guarded",

  "neutral_baseline_params": {
    "system_prompt_variant": 0,
    "few_shot_count": 0,
    "reasoning_profile": 2,
    "output_budget_bucket": 1,
    "response_schema_mode": 0,
    "tool_policy_variant": 0
  },
  "neutral_fitness":    0.425,

  "calibrated_params":  { "system_prompt_variant": 1, "reasoning_profile": 1 },
  "calibrated_fitness": 0.925,

  "uplift_absolute": 0.500,
  "uplift_percent":  117.65,

  "walk_forward": {
    "train_best_fitness": 0.925,
    "test_fitness":       0.925,
    "generalization_gap": 0.000,
    "kc4_correlation":    null,
    "passed":             true
  },

  "hard_gate_pass_rate": 1.00,
  "sensitivity_ranking": [
    { "axis": "system_prompt_variant", "gini_delta": 1.453, "raw_stress": 0.500, "rank": 0 },
    { "axis": "reasoning_profile",     "gini_delta": 1.095, "raw_stress": 0.425, "rank": 1 },
    { "axis": "few_shot_count",        "gini_delta": 0.259, "raw_stress": 0.250, "rank": 2 },
    { "axis": "output_budget_bucket",  "gini_delta": -0.935, "raw_stress": 0.000, "rank": 3 },
    { "axis": "response_schema_mode",  "gini_delta": -0.935, "raw_stress": 0.000, "rank": 4 },
    { "axis": "tool_policy_variant",   "gini_delta": -0.935, "raw_stress": 0.000, "rank": 5 }
  ],

  "n_candidates_evaluated": 22,
  "total_api_calls":        344,
  "usage_summary": {
    "input_tokens":             14262,
    "output_tokens":             4620,
    "cache_creation_input_tokens":  0,
    "cache_read_input_tokens":  15840
  },
  "status": "OK"
}
```

Notes a reader can verify from this artifact alone:

- **Sensitivity unlocked two axes.** `system_prompt_variant` (Gini 1.45) and `reasoning_profile` (Gini 1.09) are both well above the next axis (`few_shot_count` at 0.26). With `--unlock-k 2`, only those two entered the grid. Budget / schema / tool axes had zero raw stress and negative Gini (the metric reports a signed value; negative means the axis contributed no useful signal beyond the baseline's natural variance), so locking them out saved 3× the grid combinations.
- **Calibration earned +117.6% over the neutral baseline.** The neutral-baseline `fitness = 0.425` jumped to `0.925` after the search, recorded side by side on the artifact — not just the peak but the earn-over-doing-nothing.
- **Walk-forward passed.** `train = test = 0.925`, gap = 0%. KC-4 Pearson is `null` because the deterministic stub produces identical per-item scores on both slices, which collapses the correlation variance. The `WalkForward.passed` flag still fires `true` because the gap check is satisfied and KC-4 is skipped (not failed) when undefined.
- **Hard-gate pass rate is 1.0.** No `no_refusal` failures in any item. Any gate failure would have collapsed its item's fitness contribution to zero.

The real calibration engine is not mocked anywhere above. `omega_lock.run_p1` ran with its production search logic; only the *target* and *judge* layers were replaced with deterministic stubs so the output is reproducible without API access. This is the same technique the integration test in `tests/test_calibrate_integration.py` uses to catch seams between the adapter layer and the search engine.

The artifact is byte-deterministic:

```
$ md5sum examples/reference/reference_artifact.json
dedab51a32b2ab5ff462c101438cccd8  examples/reference/reference_artifact.json
```

Two consecutive runs on any machine produce the same hash. If a change to the adapter layer, the fitness function, or the artifact schema alters the output, the hash shifts and the reviewer knows exactly where to look.

### 11.3 Preflight + adaptation on a weak-infrastructure config

A second reproducibility script exercises the preflight and adaptation layer on a deliberately weak configuration — small dataset, same-vendor target and judge, homogeneous variants, concentrated rubric, noisy judge, unreliable schema parse, long projected wall time. Deterministic; no API calls.

```bash
python examples/reference/reproduce_preflight_demo.py
# writes examples/reference/reference_preflight_report.json
# writes examples/reference/reference_adaptation_plan.json
```

Analytical findings (seven traps):

```
[REAL       high   ] self_agreement_bias
    Target and judge are identical: openai/gpt-4o-mini. Judge will share the target's distributional biases.
[REAL       high   ] small_sample_kc4_power
    Test slice has 5 items. Pearson correlation at n=5 has weak statistical power; KC-4 pass/fail may be random.
[NEW        medium ] variants_homogeneous
    All 2 system prompts have near-identical length (21-29 chars); they may be too similar to produce meaningful sensitivity.
[REAL       medium ] rubric_weight_concentration
    Dimension 'accuracy' carries 85% of the rubric weight; judge noise on that single dimension will dominate fitness.
[GHOST      low    ] judge_budget_too_small
[NEW        low    ] empty_reference_with_strict_rubric
[GHOST      low    ] no_held_out_slice
```

Empirical measurements (simulated values feeding the adaptation logic):

```
judge.consistency           = 0.55
judge.anchoring_usage       = 0.40
endpoint.schema_reliability = 0.67
endpoint.context_margin     = 0.35
perf.projected_wall_time    = 5.2h
perf.noise_floor            = 0.180
```

The resulting `AdaptationPlan` carries six overrides plus a one-axis sensitivity skip:

| parameter | default | applied | reason |
|---|---|---|---|
| `min_kc4` | 0.5 | 0.6 | empirical noise floor 0.180 requires stronger Pearson |
| `max_gap` | 0.25 | 0.4 | small-sample test slice widens acceptable gap |
| `rescore_count` | 1 | 3 | judge consistency 0.55 < 0.60 — take median of 3 |
| `schema_mode` | strict_schema | json_object | STRICT_SCHEMA reliability 67% below 90% — fallback with post-parse validation |
| `judge_ensemble_shift` | 0.0 | 0.40 | judge consistency 0.55 — raise RuleJudge weight to 40% |
| `unlock_k` | 3 | 2 | projected wall-time 5.2h exceeds 4h — reduce unlock_k |
| `skip_axes` | `[]` | `["system_prompt_variant"]` | variants homogeneous finding |
| `preserves_discipline` | `True` | `True` | invariant never toggled off |

Every override is *monotonic toward stricter* validation: `min_kc4` only rises, `max_gap` rises only to accommodate the variance the small sample actually has, `unlock_k` only falls, `rescore_count` only rises. A plan that attempted a weaker threshold than the caller's default would be clipped at `apply_adaptation_plan` time (Appendix C invariant 10). The resulting run is still a valid calibration — just one tuned to the infrastructure's actual noise floor rather than the default assumption of a frontier-tier judge on a large dataset.

---

## 12. Validation

The test suite runs with `pytest -q` in under one second on a laptop and issues **zero live API calls**. The current head of `main` passes **149 tests** (wall time ~0.4s). Every adapter test uses `SimpleNamespace` or `MagicMock` in place of an SDK client, and asserts the *exact* shape of the outgoing request payload (model, messages, cache headers, `response_format`, reasoning directives, few-shot ordering). The sub-tool repositories `mini-omega-lock` and `mini-antemortem-cli` carry their own test suites covering probe execution and analytical trap classification respectively.

| Module | Coverage summary |
|---|---|
| `domain/` | `PromptVariants` / `MetaAxisSpace` / `CalibrationArtifact` / `Dataset` / enums / profiles — required fields, range validation, JSON roundtrip, ordinal clamping, compat-key mapping, `model_post_init` synchronisation between `best_params` and `calibrated_params`. |
| `core/fitness` | `CompositeFitness` over empty / all-pass / partial-fail / all-fail batches; per-item breakdown preserved for reporting. |
| `core/walkforward` | Pearson correlation over shared ids; zero-variance skip (kc4=None); gap arithmetic; gate pass/fail logic for both thresholds. |
| `core/sensitivity` | Gini-coefficient ranking; top-K unlock; edge cases (zero-delta axes, single-point probes). |
| `core/artifact` | Round-trip through JSON on disk; `model_post_init` invariants on load. |
| `core/profiles` | `policy_for(GUARDED/EXPEDITION)` returns distinct defaults; `relaxed_safeguards_for(...)` reports crossings. |
| `core/risk` | `assess_run_risk(...)` across OK / KC-4 fail / hard-gate fail / capability-event scenarios. |
| `providers/` | Factory rejects unknown names; respects `base_url`; lists `anthropic` / `openai` / `gemini` / `ollama`. Anthropic adapter: freeform uses `messages.create` with thinking config when reasoning enabled, omits it when OFF; strict schema uses `messages.parse`; refusal raises; JSON-object mode adds system-prompt suffix. OpenAI adapter: same paths on `chat.completions.create` / `beta.chat.completions.parse`; `reasoning_effort` rejected-retry records `CapabilityEvent`; `prompt_tokens_details.cached_tokens` normalised to `cache_read_input_tokens`; content-filter finish reason raises; missing `parsed` raises. `ollama` alias reports tier `tier_3_local`, `supports_llm_judge=False`, `experimental=True`. |
| `judges/` | `RuleJudge` (no_refusal / non_empty / json_object / regex / duplicate-check rejection / missing-check raise); `LLMJudge` (strict-schema dispatch, payload composition, non-`JudgeResult` response raise, guarded-mode ship-grade judge refusal); `EnsembleJudge` (rule-first short-circuit, LLM escalation on rule-pass, merged gate_results, non-`RuleJudge` rejection). |
| `targets/` | `PromptTarget` end-to-end with mocked provider + judge; meta-axis resolution and clamping for out-of-range inputs; usage accumulation across evaluations; `evaluation_history` retention; latency measurement; degraded-capability propagation. |
| `commands/` | CLI help lists `calibrate` / `report` / `diff`; `--version` shows `omegaprompt`; `report` renders schema-v2.0 artifacts; `diff` detects regressions on fitness, cost ratios, latency ratios, boundary-crossing flips. |
| `preflight/` | **Plugin interface only** — no probe or classifier code inside `omegaprompt`. `contracts`: severity ordering, status enum, `PreflightReport.worst_severity` / `any_real_or_new`, Pydantic `extra="forbid"` enforcement; bounds on `JudgeQualityMeasurement.consistency` (0..1), `EndpointMeasurement.schema_reliability` (0..1). `adaptation`: noise-adaptive `min_kc4` across four thresholds; consistency-driven `rescore_count`; schema-fallback trigger; wall-time-driven `unlock_k` reduction; small-sample gap widening; variant-skip axis marking; `apply_adaptation_plan` invariants (never weakens `min_kc4`, never widens `max_gap`, never raises `unlock_k`). Sub-tool probe + classifier implementations (with their own test suites) live in the `mini-omega-lock` and `mini-antemortem-cli` repositories. |
| `test_calibrate_integration.py` | **Drives the real `omega_lock.run_p1`** with a deterministic in-memory `CalibrableTarget` (no mocks on the search engine). Asserts the artifact's `calibrated_params`, `neutral_baseline_params`, `walk_forward`, and `sensitivity_ranking` match `P1Result`'s actual shape — the regression that this test catches is drift between the adapter layer and the search engine that per-module unit tests cannot reach. |

Run with `uv run pytest -q` (or `python -m pytest -q`). The wall-clock time is dominated by Pydantic model compilation on first import.

---

## 13. Comparative positioning

| Approach | What it does well | What `omegaprompt` adds |
|---|---|---|
| **promptfoo** | Runs prompts against test cases with assertion-based grading. | Pre-declared walk-forward gate, sensitivity-ranked axis unlock, `hard_gate × soft_score` fitness, machine-readable diffable artifact. Composable — promptfoo-style assertions plug in as `RuleJudge` checks. |
| **DSPy** | Prompt synthesis via program abstraction + bootstrapped few-shot. | Orthogonal concern. DSPy *produces* candidate prompts; `omegaprompt` *decides which one ships* after walk-forward validation. DSPy outputs plug in as `system_prompts` entries in `PromptVariants`. |
| **Optuna / Ray Tune on prompts** | Generic hyperparameter optimisation. | Walk-forward ship gate and pre-declared kill criteria out of the box; schema-enforced LLM-as-judge via each vendor's native parse path; provider-neutral meta-axes; explicit `CalibrationArtifact` schema CI can diff. |
| **Provider-native evaluation dashboards** | Rubric-based grading inside one vendor's console. | Cross-vendor judging (break self-agreement bias); local artifact that does not require vendor login; deterministic `diff` for regression detection; `expedition` mode for controlled boundary crossing. |
| **Hand-rolled eval scripts** | Fast to author for a single workload. | Structured data contract (`Dataset` / `JudgeRubric` / `PromptVariants` / `CalibrationArtifact`); capability-tier policy; pre-declared gates that cannot be lowered after the fact; CI integration without bespoke glue. |

The unique selling point is *discipline over search*. The search engine is [`omega-lock`](https://github.com/hibou04-ops/omega-lock), which was shipped and validated against a different domain (parameter calibration in quantitative trading) before this prompt adapter was written. `omegaprompt` contributes the prompt-specific adapter, three provider-neutral judges, the hard-gates-first fitness shape, and the capability / profile / artifact architecture.

---

## 14. Limitations and scope boundaries

### Not a safety evaluator

`no_safety_violation` can be declared as a hard gate, but the judge is a rubric-scored LLM, not a trained safety classifier. For regulated safety evaluation, pair `omegaprompt` with a dedicated safety eval suite (AILuminate, HELM, vendor-specific red-team harnesses).

### Not a replacement for production telemetry

Offline calibration on a curated dataset is a cheap screening step. Real-traffic A/B with business metrics remains the ground truth. `omegaprompt` makes the offline step disciplined; it does not replace online evaluation.

### Not a benchmark of vendor capability

The fitness numbers reflect *your rubric applied to your dataset on the models you configured*. They are not benchmarks of absolute model capability and are not portable to other domains without re-validation.

### Judge drift is a real concern

The LLM judge's scoring distribution can drift across model releases. The v1.1 roadmap includes a multi-judge validation pattern (`judge_v1` vs `judge_v2` on the top-K) so disagreement becomes a trust signal rather than a silent failure.

### Cost is non-trivial

A typical run (10-item dataset, 125-candidate grid, walk-forward) on frontier-tier cloud providers costs in the tens of dollars. Mitigations: cheaper judge during iteration, prompt-cache-aware scheduling within a 5-minute window, local target via Ollama when quality permits.

### Not all providers are ship-grade judges

Guarded mode blocks local providers in the judge position by policy. The policy can be overridden in expedition mode (with the relaxation recorded) or by adjusting the provider's capability declaration if you have independently validated its judge quality on your domain.

---

## 15. Roadmap

**Shipped (v1.0)**
- Provider-neutral meta-axes (`reasoning_profile`, `output_budget_bucket`, `response_schema_mode`, `tool_policy_variant`).
- Unified `LLMProvider.call(ProviderRequest) -> ProviderResponse` + `capabilities()` contract.
- `ExecutionProfile` (guarded / expedition) + structural risk reporting.
- `CalibrationArtifact` schema v2.0 (neutral baseline vs calibrated, capability events, boundary warnings, ship recommendation).
- `RuleJudge` / `LLMJudge` / `EnsembleJudge`.
- `gemini` placeholder + `ollama` / `local` / `vllm` / `llama_cpp` local adapter family.
- CLI: `calibrate` / `report` / `diff`. Backward-compat `omegaprompt` alias.
- Integration test against real `omega_lock.run_p1`.

**v1.1 (judge trust + tooling depth)**
- Native Gemini adapter (replace placeholder).
- Multi-judge validation pattern: `judge_v1` + `judge_v2` over top-K; disagreement as a first-class trust signal.
- `--dry-run` with cost estimate before launching.
- Additional rule-gate predicates (language detection, length bounds, schema validation against a supplied JSON Schema).

**v1.2 (ecosystem)**
- Benchmark harness: multi (task × rubric × seed) scorecards.
- GitHub Action for CI regression gating via `omegaprompt diff`.
- HTML report rendering (`omegaprompt report --format html`).
- Native HuggingFace Inference adapter.

**Explicitly out of scope**
- Hosted dashboard, database-backed history, multi-tenant service. `omegaprompt` is a local developer tool. Keep it local.

Full changelog: [CHANGELOG.md](CHANGELOG.md).

---

## 16. Prior art and credits

- **Train / test split with a pre-declared gate.** The foundational ML defence against overfitting, documented in every undergraduate curriculum. The specific implementation here (Pearson rank correlation threshold, pre-declared and unmodifiable) is `omega-lock`'s KC-4 kill criterion.
- **LLM-as-judge.** Pattern formalised in *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (Zheng et al., 2023). `omegaprompt` implements the pattern with schema enforcement at the SDK boundary (Pydantic via each vendor's native parse path) so malformed judge responses raise before polluting the fitness.
- **Winchester defence.** A quant-finance discipline: *kill criteria declared before the run cannot be relaxed after.* Used here to argue that `--max-gap` and `--min-kc4` must be enforced in configuration, not retroactively tuned on inspection of scores.
- **Sensitivity-driven coordinate descent.** Stress measurement and top-K unlock are the parameter-calibration primitives introduced by `omega-lock` (v0.1.4), originally for trading-strategy calibration, ported here to prompt configuration.
- **Antemortem discipline.** The pre-implementation reconnaissance methodology under which this project was designed and built. Every non-trivial change runs through [`antemortem-cli`](https://github.com/hibou04-ops/antemortem-cli) before the first keystroke. The case studies in the [methodology repository](https://github.com/hibou04-ops/Antemortem) record the recon for this codebase.

Naming: *omega-lock* (parameter calibration) → *omegaprompt* (prompt calibration). The family branding is intentional. `omega-lock` was extracted from a trading-strategy calibration that ended in `KC-4 FAIL` — the overfitting defence firing exactly as designed. `omegaprompt` is the same defence applied one layer up, and the sub-tools `mini-omega-lock` / `mini-antemortem-cli` extend the pattern to preflight measurement.

---

## Appendix A: data contracts

Pydantic v2 models, `extra="forbid"` unless noted.

```python
# domain/dataset.py

class DatasetItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    input: str
    reference: str | None = None
    metadata: dict = {}

class Dataset(BaseModel):
    items: list[DatasetItem]
    @classmethod
    def from_jsonl(cls, path) -> Dataset: ...


# domain/params.py

class PromptVariants(BaseModel):
    system_prompts: list[str]                        # non-empty
    few_shot_examples: list[dict[str, str]] = []     # {input, output}

class MetaAxisSpace(BaseModel):
    system_prompt_idx_max: int                        # ge=0
    few_shot_min: int = 0
    few_shot_max: int = 3                             # >= few_shot_min
    reasoning_profiles:   list[ReasoningProfile]     # non-empty
    output_budgets:       list[OutputBudgetBucket]   # non-empty
    response_schema_modes: list[ResponseSchemaMode]  # non-empty
    tool_policy_variants: list[ToolPolicyVariant]    # non-empty

class ResolvedPromptParams(BaseModel):
    system_prompt_variant: int
    few_shot_count: int
    reasoning_profile: ReasoningProfile = STANDARD
    output_budget_bucket: OutputBudgetBucket = MEDIUM
    response_schema_mode: ResponseSchemaMode = FREEFORM
    tool_policy_variant: ToolPolicyVariant = NO_TOOLS


# domain/judge.py

class Dimension(BaseModel):
    name: str
    description: str
    weight: float                                    # ge=0
    scale: tuple[int, int] = (1, 5)                  # hi > lo

class HardGate(BaseModel):
    name: str
    description: str
    evaluator: Literal["judge", "rule", "post"] = "judge"

class JudgeRubric(BaseModel):
    dimensions: list[Dimension]                      # min_length=1, unique names, sum(weight) > 0
    hard_gates: list[HardGate] = []                  # unique names

class JudgeResult(BaseModel):
    scores: dict[str, int]                           # keyed by dimension name
    gate_results: dict[str, bool] = {}               # keyed by gate name
    notes: str = ""


# domain/profiles.py

class ExecutionProfile(str, Enum):
    GUARDED = "guarded"
    EXPEDITION = "expedition"

class ShipRecommendation(str, Enum):
    SHIP = "ship"
    HOLD = "hold"
    ROLLBACK = "rollback"

class RiskCategory(str, Enum): ...
class BoundaryWarning(BaseModel): ...
class RelaxedSafeguard(BaseModel): ...


# providers/base.py

class CapabilityTier(str, Enum):
    CORE = "tier_1_core_parity"
    CLOUD = "tier_2_cloud_grade"
    LOCAL = "tier_3_local"

class CapabilityEvent(BaseModel):
    capability: str
    requested: str
    applied: str
    reason: str
    user_visible_note: str
    affects_guarded_boundary: bool = True

class ProviderCapabilities(BaseModel):
    provider: str
    tier: CapabilityTier
    supports_strict_schema: bool = False
    supports_json_object: bool = False
    supports_reasoning_profiles: bool = False
    supports_usage_accounting: bool = True
    supports_llm_judge: bool = False
    ship_grade_judge: bool = False
    supports_tools: bool = False
    experimental: bool = False
    placeholder: bool = False
    notes: list[str] = []

class ProviderRequest(BaseModel):
    system_prompt: str
    user_message: str
    few_shots: list[dict[str, str]] = []
    reasoning_profile: ReasoningProfile = STANDARD
    output_budget_bucket: OutputBudgetBucket = MEDIUM
    response_schema_mode: ResponseSchemaMode = FREEFORM
    tool_policy_variant: ToolPolicyVariant = NO_TOOLS
    execution_profile: ExecutionProfile = GUARDED
    output_schema: type[BaseModel] | None = None     # required when STRICT_SCHEMA

class ProviderResponse(BaseModel):
    text: str = ""
    parsed: BaseModel | None = None
    usage: dict[str, int] = {}
    finish_reason: str | None = None
    latency_ms: float = 0.0
    degraded_capabilities: list[CapabilityEvent] = []
    capability_notes: list[str] = []


# domain/result.py

class EvalItemResult(BaseModel):
    item_id: str
    params: dict
    raw_output: str
    judge: JudgeResult
    token_usage: dict[str, int] = {}
    latency_ms: float = 0.0
    degraded_capabilities: list[CapabilityEvent] = []
    boundary_warnings: list[BoundaryWarning] = []

class EvalResult(BaseModel):
    params: dict
    resolved_params: dict = {}
    item_results: list[EvalItemResult]
    fitness: float
    n_trials: int
    hard_gate_pass_rate: float = 0.0
    usage_summary: dict[str, int] = {}
    latency_ms: float = 0.0
    estimated_cost_units: float = 0.0
    degraded_capabilities: list[CapabilityEvent] = []
    boundary_warnings: list[BoundaryWarning] = []
    within_guarded_boundaries: bool = True
    ship_recommendation: ShipRecommendation = HOLD
    metadata: dict = {}

class WalkForwardResult(BaseModel):
    train_best_fitness: float
    test_fitness: float
    generalization_gap: float
    kc4_correlation: float | None
    passed: bool

class CalibrationArtifact(BaseModel):
    schema_version: str = "2.0"
    engine_name: str = "omegaprompt"
    method: str
    unlock_k: int                                    # ge=0
    selected_profile: ExecutionProfile = GUARDED
    neutral_baseline_params: dict = {}
    calibrated_params: dict = {}
    neutral_fitness: float = 0.0
    calibrated_fitness: float = 0.0
    uplift_absolute: float = 0.0
    uplift_percent: float = 0.0
    quality_per_cost_neutral: float = 0.0
    quality_per_cost_best: float = 0.0
    quality_per_latency_neutral: float = 0.0
    quality_per_latency_best: float = 0.0
    boundary_warnings: list[BoundaryWarning] = []
    degraded_capabilities: list[CapabilityEvent] = []
    ship_recommendation: ShipRecommendation = HOLD
    stayed_within_guarded_boundaries: bool = True
    additional_uplift_from_boundary_crossing: float = 0.0
    relaxed_safeguards: list[RelaxedSafeguard] = []
    guarded_boundary_crossed: bool = False
    cost_basis: str = "normalized_token_units"
    best_params: dict                                # kept for v1.x compat
    best_fitness: float
    walk_forward: WalkForwardResult | None = None
    hard_gate_pass_rate: float                       # 0..1
    sensitivity_ranking: list[dict] = []
    n_candidates_evaluated: int
    total_api_calls: int
    usage_summary: dict[str, int] = {}
    latency_summary_ms: dict[str, float] = {}
    target_provider: str | None = None
    target_model: str | None = None
    judge_provider: str | None = None
    judge_model: str | None = None
    target_capabilities: ProviderCapabilities | None = None
    judge_capabilities: ProviderCapabilities | None = None
    status: str = "OK"                               # OK / FAIL_KC4_GATE / FAIL_HARD_GATES / FAIL_NO_CANDIDATES
    rationale: str = ""
```

`ResolvedPromptParams` and `ProviderRequest` carry `@model_validator(mode="before")` compat mappings that accept the v1.0 axis names (`system_prompt_idx`, `output_budget`, `tool_policy`) and rewrite them to the v1.1+ canonical names (`system_prompt_variant`, `output_budget_bucket`, `tool_policy_variant`). Read-side `@property` accessors let either name be used.

---

## Appendix B: meta-axis to vendor-parameter mapping

Canonical translation table, excerpted from the adapter implementations.

| Meta-axis value | Anthropic | OpenAI / compatible | Local (Ollama / vLLM / llama.cpp) |
|---|---|---|---|
| `reasoning_profile = OFF` | no `thinking` block | no `reasoning_effort` | system prompt unchanged |
| `reasoning_profile = LIGHT` | `thinking={type:adaptive}` + `effort: low` | `reasoning_effort: low` (if supported) | system-prompt suffix: "think briefly" |
| `reasoning_profile = STANDARD` | `thinking={type:adaptive}` + `effort: medium` | `reasoning_effort: medium` | system-prompt suffix: "think step by step" |
| `reasoning_profile = DEEP` | `thinking={type:adaptive}` + `effort: high` | `reasoning_effort: high` | system-prompt suffix: "think carefully step by step" |
| `output_budget_bucket = SMALL` | `max_tokens=1024` | `max_tokens=1024` | `max_tokens=1024` |
| `output_budget_bucket = MEDIUM` | `max_tokens=4096` | `max_tokens=4096` | `max_tokens=4096` |
| `output_budget_bucket = LARGE` | `max_tokens=16000` | `max_tokens=16000` | `max_tokens=16000` |
| `response_schema_mode = FREEFORM` | `messages.create` | `chat.completions.create` | `chat.completions.create` |
| `response_schema_mode = JSON_OBJECT` | `messages.create` + system-prompt JSON suffix | `response_format={type:json_object}` | best-effort system-prompt instruction |
| `response_schema_mode = STRICT_SCHEMA` | `messages.parse(output_format=T)` | `beta.chat.completions.parse(response_format=T)` | not supported; guarded mode raises |
| `tool_policy_variant = NO_TOOLS` | no `tools` argument | no `tools` argument | no `tools` argument |
| `tool_policy_variant = TOOL_OPTIONAL` | `tools=[...]`, no `tool_choice` | `tools=[...], tool_choice="auto"` | adapter-specific |
| `tool_policy_variant = TOOL_REQUIRED` | `tools=[...], tool_choice={type:"any"}` | `tools=[...], tool_choice="required"` | adapter-specific |

Any cell that reads "not supported" or "best-effort" emits a `CapabilityEvent` at runtime and, under guarded mode, may block the run depending on the execution profile policy.

---

## Appendix C: invariants

The following properties hold by construction and are enforced either in the Pydantic schema layer or in a dedicated test. They are the "theorems" a reviewer can rely on without reading the implementation.

1. **No client-side schema regex.** `STRICT_SCHEMA` mode always dispatches to the vendor's native parse path (`messages.parse` on Anthropic, `beta.chat.completions.parse` on OpenAI). A malformed structured response raises `ValidationError` before the calibration loop sees it.
2. **Hard-gate fitness collapse.** For any `(item, params)` pair, if any `hard_gate` returns `False`, the item's contribution to `CompositeFitness` is `0.0`. No soft penalty, no partial credit.
3. **Walk-forward threshold immutability.** `--max-gap` and `--min-kc4` are CLI arguments resolved once at run start and recorded on the artifact. There is no API surface for modifying them mid-run.
4. **Capability-event propagation.** Every `CapabilityEvent` emitted in `ProviderResponse.degraded_capabilities` flows up through `EvalItemResult` → `EvalResult` → `CalibrationArtifact.degraded_capabilities` unchanged. An adapter cannot silently degrade.
5. **Guarded-profile ship-grade judge check.** Under `ExecutionProfile.GUARDED`, `LLMJudge.score` raises `JudgeError` if `provider.capabilities().supports_llm_judge` is `False`. No implicit waiver.
6. **Deterministic decision derivation.** `ship_recommendation`, `status`, `stayed_within_guarded_boundaries`, and `guarded_boundary_crossed` are computed by pure functions of the artifact fields and the profile policy. Same input, same output.
7. **Backward-compat key rewrite is lossless.** `ProviderRequest` and `ResolvedPromptParams` accept legacy keys (`system_prompt_idx`, `output_budget`, `tool_policy`) via a `@model_validator(mode="before")` that rewrites to the canonical names. Read-side `@property` accessors preserve both names.
8. **`neutral_baseline_params` and `calibrated_params` never contradict `best_params`.** `CalibrationArtifact.model_post_init` synchronises `best_params ↔ calibrated_params` and `best_fitness ↔ calibrated_fitness` when one side is missing, so downstream consumers can read either pair without an existence check.
9. **Artifact JSON is stable across runs with identical inputs.** The reproducibility script in §11.2 demonstrates this empirically; the integration test in `tests/test_calibrate_integration.py` enforces it in CI.
10. **Adaptation only strengthens the discipline.** `apply_adaptation_plan` uses `max(default_min_kc4, plan_override)`, `min(default_max_gap, plan_override)`, and `min(default_unlock_k, plan_override)`. A plan that attempted to widen tolerance is clipped to the caller's configuration. See §5.8 and Appendix D.

---

## Appendix D: AdaptationPlan contract

```python
class ParameterOverride(BaseModel):
    parameter: str
    default: Any
    applied: Any
    reason: str

class AdaptationPlan(BaseModel):
    # Walk-forward gate
    min_kc4_override: float | None = None          # never lower than caller default
    max_gap_override: float | None = None          # never higher than caller default

    # Search
    unlock_k_override: int | None = None           # never larger than caller default
    skip_axes: list[str] = []                      # axes pre-excluded from sensitivity

    # Evaluation
    rescore_count: int = 1                         # N-judge median when consistency low
    rubric_weight_overrides: dict[str, float] = {} # zero out unreliable dims only
    schema_mode_fallback: ResponseSchemaMode | None = None   # STRICT -> JSON_OBJECT only
    judge_ensemble_shift: float | None = None      # extra weight toward RuleJudge

    # Scheduling
    candidate_budget_cap: int | None = None
    dataset_reorder_for_cache: bool = False

    # Audit
    overrides: list[ParameterOverride] = []        # itemised reason trail
    rationale: list[str] = []                      # human-readable summary
    preserves_discipline: bool = True              # must be True by construction
```

### Derivation rules (deterministic)

Same `PreflightReport` in → same `AdaptationPlan` out. The rules are captured in `omegaprompt.preflight.adaptation.derive_adaptation_plan` and unit-tested across noise levels, consistency levels, schema-reliability levels, wall-time projections, and analytical-finding combinations.

### Application rules (invariant-preserving)

```python
def apply_adaptation_plan(
    plan: AdaptationPlan,
    *,
    min_kc4: float,
    max_gap: float,
    unlock_k: int,
) -> tuple[float, float, int]:
    applied_kc4   = max(min_kc4,  plan.min_kc4_override  or min_kc4)
    applied_gap   = min(max_gap,  plan.max_gap_override  or max_gap)
    applied_unlock = min(unlock_k, plan.unlock_k_override or unlock_k)
    return applied_kc4, applied_gap, applied_unlock
```

Three clipping rules encode a single principle: **an AdaptationPlan may tighten, never loosen**. A malicious or buggy plan that tries to lower `min_kc4` from 0.6 to 0.4 is silently clipped to 0.6 — the caller's strictness wins.

### Audit trail

Every override carries its `parameter`, `default`, `applied`, and `reason`. The plan's `rationale` is a free-form list of human-readable one-liners. Both are serialised into the `CalibrationArtifact` so that a reviewer reading an artifact six months after the run can see not only what parameters were used but *why* they diverged from the defaults.

### Sub-unit boundary

`omegaprompt.preflight` ships the **contract + adaptation logic only**. Probe execution (`mini-omega-lock`) and analytical classification (`mini-antemortem-cli`) live in separate repositories / PyPI packages that depend on this contract. An external sub-tool only needs to emit a `PreflightReport` conforming to :mod:`omegaprompt.preflight.contracts` for `derive_adaptation_plan` to turn it into an `AdaptationPlan` the main pipeline consumes. Standalone users install nothing extra; the preflight interface is a no-op surface for them.

---

## Citing

Short form:

```
omegaprompt v1.0.0 — provider-neutral prompt calibration engine.
https://github.com/hibou04-ops/omegaprompt, 2026.
```

BibTeX:

```bibtex
@software{omegaprompt_2026,
  author  = {hibou04-ops},
  title   = {{omegaprompt}: Provider-neutral prompt calibration engine
             with sensitivity-ranked meta-axes, walk-forward ship gates,
             and structural capability reporting},
  version = {1.0.0},
  year    = {2026},
  url     = {https://github.com/hibou04-ops/omegaprompt}
}

@software{omegalock_2026,
  author  = {hibou04-ops},
  title   = {{omega-lock}: Sensitivity-driven coordinate-descent
             calibration framework with walk-forward validation and
             pre-declared kill criteria},
  version = {0.1.4},
  year    = {2026},
  url     = {https://github.com/hibou04-ops/omega-lock}
}

@software{antemortem_2026,
  author  = {hibou04-ops},
  title   = {{Antemortem}: AI-assisted pre-implementation reconnaissance
             for software changes with disk-verified citations},
  version = {0.1.1},
  year    = {2026},
  url     = {https://github.com/hibou04-ops/Antemortem}
}
```

---

## License

Apache 2.0. See [LICENSE](LICENSE).

## Colophon

Designed, implemented, and shipped solo. Adapter layer over `omega-lock`; zero calibration-engine reimplementation. Every non-trivial change is pre-authored through `antemortem-cli`'s recon discipline. Tests run offline; no live API calls in CI.
