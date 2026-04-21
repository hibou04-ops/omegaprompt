# omegaprompt

[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)
[![PyPI](https://img.shields.io/badge/pypi-1.0.0-blue.svg)](https://pypi.org/project/omegaprompt/)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#status--roadmap)
[![Providers](https://img.shields.io/badge/providers-anthropic%20%7C%20openai%20%7C%20openai--compatible-informational.svg)](#provider-support)
[![Parent framework](https://img.shields.io/badge/framework-omega--lock-blueviolet.svg)](https://github.com/hibou04-ops/omega-lock)

> **Your prompt scores 4.8/5 on the examples you picked. It collapses on day 2 of prod.**
>
> This failure has a name in machine learning — *overfitting* — and a solution that is older than LLMs themselves: a train/test split with a pre-declared gate. omegaprompt is that gate for prompt engineering, model-agnostic by design. Target and judge each speak to a pluggable `LLMProvider` (Anthropic, OpenAI, or any OpenAI-compatible endpoint including local Ollama). If the training-best prompt doesn't rank on held-out data, it fails KC-4 and nothing ships.

```bash
pip install omegaprompt
```

한국어 README: [README_KR.md](README_KR.md)

---

## Table of Contents

- [The failure mode this solves](#the-failure-mode-this-solves)
- [Worked example: overfit detection in action](#worked-example-overfit-detection-in-action)
- [The calibratable axes](#the-calibratable-axes)
- [Provider support](#provider-support)
- [Cross-vendor validation](#cross-vendor-validation)
- [The data contract](#the-data-contract)
- [Architecture](#architecture)
- [Design decisions worth defending](#design-decisions-worth-defending)
- [What this is NOT](#what-this-is-not)
- [Cost & performance](#cost--performance)
- [Validation](#validation)
- [The 3-layer stack](#the-3-layer-stack)
- [Relation to adjacent tools](#relation-to-adjacent-tools)
- [FAQ for skeptics](#faq-for-skeptics)
- [Prior art & credit](#prior-art--credit)
- [Status & roadmap](#status--roadmap)
- [Contributing](#contributing)
- [Citing](#citing)
- [License](#license)

---

## The failure mode this solves

Prompt engineering has a predictable failure mode. You iterate on a handful of hand-picked examples. The scores go up. The prompt looks great. You ship it. Two days later, it falls over on an input you never imagined.

This is not a prompt-engineering skill issue. It is the same failure that drove the entire train/test/validate split into mainstream ML in the 1990s: **you cannot validate a learned configuration on the data you learned from.** Every tutorial in every ML course starts here. Every prompt-engineering guide skips it.

omegaprompt ports the ML defense to the prompt setting, via its sibling project [omega-lock](https://github.com/hibou04-ops/omega-lock). Three ideas transfer verbatim:

1. **Sensitivity measurement.** Which prompt axes actually matter? Perturb each one around a neutral baseline, rank by Gini coefficient of the fitness delta. Stop spending search budget on axes that don't move the score.
2. **Top-K unlock, lock the rest.** Search only in the subspace that moves fitness. The rest stay at neutral.
3. **Walk-forward validation with a pre-declared gate.** After the search finishes on the training slice, re-evaluate on a held-out test slice the searcher never saw. Require Pearson correlation above a pre-declared threshold (`KC-4`) before the result ships. No tuning the threshold after the fact.

Two guardrails separate omegaprompt from *"ask a judge and pick the highest score"*:

1. **Hard gates collapse fitness to zero.** `no_refusal`, `format_valid`, `no_safety_violation` — any gate fails on an item, that item contributes zero. A prompt that scores 5.0 on ten tasks and refuses on the eleventh does *not* rank above one that consistently scores 4.2 with no refusals. Soft penalties reward prompts that *almost* refuse. Hard zeros don't.
2. **Walk-forward is the ship gate.** omega-lock's KC-4 (Pearson correlation between train and test rankings) is pre-declared in config and enforced mechanically. You cannot retroactively lower the threshold to rescue a borderline candidate.

---

## Worked example: overfit detection in action

The canonical failure: you iterate prompts against a 10-example training set and pick the winner.

```
Candidate prompt A:  train_fitness = 0.923  (4.6/5 average)
Candidate prompt B:  train_fitness = 0.876  (4.4/5 average)
```

Prompt A wins on training. Ship A, right? Walk-forward on a 10-example test slice:

```
Candidate prompt A:  train = 0.923  test = 0.612  gen_gap = 33.7%
Candidate prompt B:  train = 0.876  test = 0.841  gen_gap =  4.0%

omega-lock run_p1 status: FAIL:KC-4
Reason: spearman(train_ranks, test_ranks) = 0.18 < 0.30 threshold
Candidate A's train-ranking is uncorrelated with its test-ranking. Do not ship.
```

Prompt A overfit the training slice. It found a style that flattered the judge's surface features on the 10 examples you happened to pick — which is exactly what the ML failure mode predicts. omegaprompt's calibration surfaced this mechanically; prompt B would have been the correct ship decision.

Without KC-4, you ship A and discover the gap in prod. With KC-4, the calibration fails loud and you find the gap before deploy.

The generalization gap is recorded in the output artifact:

```json
{
  "schema_version": "1.0",
  "method": "p1",
  "unlock_k": 3,
  "best_params": {"system_prompt_idx": 2, "few_shot_count": 1, "reasoning_profile": "deep"},
  "best_fitness": 0.876,
  "walk_forward": {
    "train_best_fitness": 0.876,
    "test_fitness": 0.841,
    "generalization_gap": 0.040,
    "kc4_correlation": 0.84,
    "passed": true
  },
  "hard_gate_pass_rate": 1.00,
  "sensitivity_ranking": [
    {"axis": "system_prompt_idx", "gini_delta": 0.52, "rank": 0},
    {"axis": "few_shot_count",    "gini_delta": 0.33, "rank": 1}
  ],
  "status": "OK",
  "target_provider": "openai",
  "target_model": "gpt-4o",
  "judge_provider": "anthropic",
  "judge_model": "claude-opus-4-7"
}
```

This is the shape of every `CalibrationArtifact` this tool produces. Machine-readable, diffable across prompt revisions, trivial to gate on in CI — `omegaprompt diff old.json new.json` exits non-zero on regression.

---

## The calibratable axes

`PromptTarget` exposes six **provider-neutral meta-axes** to the searcher. Each axis captures a semantic dimension of prompt configuration; every provider adapter maps them to its vendor's native parameters internally. The public contract carries no vendor knobs.

| Axis | Type | Meaning |
|---|---|---|
| `system_prompt_idx` | int | Index into your pool of candidate system prompts (`PromptVariants.system_prompts`). |
| `few_shot_count` | int | How many examples to include from `PromptVariants.few_shot_examples`. 0 = zero-shot. |
| `reasoning_profile` | enum | `off / light / standard / deep`. Maps to Anthropic's adaptive thinking + `effort`, OpenAI's `reasoning_effort`, or a system-prompt suffix on vendors without native reasoning knobs. |
| `output_budget` | enum | `small / medium / large`. Resolves to the vendor's `max_tokens` (1024 / 4096 / 16000). |
| `response_schema_mode` | enum | `freeform / json_object / strict_schema`. Dispatches to the vendor's native parse path when STRICT. |
| `tool_policy` | enum | `no_tools / tool_optional / tool_required`. No-op for plain chat targets; surfaces when the target exposes tool schemas. |

The `MetaAxisSpace` declarative object lets you lock axes out (pass a single-member list) when some dimensions are pre-decided. Typical calibration leaves three axes open and unlocks the top-K by sensitivity. The artifact records enum values, not integer indices — `"reasoning_profile": "deep"` is self-describing regardless of which vendor produced it.

> **v1.0 note.** The v0.2 axes (`effort_idx`, `thinking_enabled`, `max_tokens_bucket`) were named after Anthropic's API contract. They are replaced by the provider-neutral enums above. The v0.2 `PromptSpace` is aliased to `MetaAxisSpace` for import-path compatibility, but the field set changed — see CHANGELOG for the v0.2 → v1.0 migration guide.

---

## Provider support

omegaprompt has two LLM-call boundaries — the **target** (the prompt under calibration, free-form output) and the **judge** (rubric-based scoring, schema-enforced output). Both go through an `LLMProvider` Protocol. Each adapter uses its vendor's strongest native paths — no client-side JSON regex parsing anywhere in the pipeline.

| Provider | Flag (target) | Flag (judge) | Default model | Env var | Native paths |
|---|---|---|---|---|---|
| Anthropic | `--target-provider anthropic` | `--judge-provider anthropic` | `claude-opus-4-7` | `ANTHROPIC_API_KEY` | `messages.create` / `messages.parse` with explicit `cache_control` |
| OpenAI | `--target-provider openai` | `--judge-provider openai` | `gpt-4o` | `OPENAI_API_KEY` | `chat.completions.create` / `beta.chat.completions.parse` |
| OpenAI-compatible | Same + `--target-base-url <url>` | Same + `--judge-base-url <url>` | user-supplied via `--*-model` | `OPENAI_API_KEY` (or any string on unauthenticated local) | Same as OpenAI |

Compatible endpoints covered via `--base-url`: **Azure OpenAI**, **Groq**, **Together.ai**, **OpenRouter**, and **local Ollama** at `http://localhost:11434/v1`. Adding a new provider is one module that satisfies the `LLMProvider` Protocol (one method: `call(ProviderRequest) -> ProviderResponse`) plus a line in the factory registry.

**The Protocol** (`src/omegaprompt/providers/base.py`):

```python
class LLMProvider(Protocol):
    name: str
    model: str

    def call(self, request: ProviderRequest) -> ProviderResponse: ...
```

`ProviderRequest` carries the meta-axes (`reasoning_profile`, `output_budget`, `response_schema_mode`, `tool_policy`) as enums. Each adapter translates them to vendor-native parameters — `reasoning_profile=deep` becomes `thinking={"type":"adaptive"}` + `effort="high"` on Anthropic, `reasoning_effort="high"` on OpenAI, a system-prompt suffix on plain chat endpoints. One method. No SDK leakage. Same meta-axes search across every vendor.

---

## Cross-vendor validation

The single strongest configuration this tool enables is **target on one provider, judge on another.**

Why it matters: if target and judge are the same model (or same vendor), self-agreement bias contaminates the score. A weak OpenAI output graded by an OpenAI judge may still look acceptable because the judge shares the target's biases. Grade that output with an Anthropic judge — or vice-versa — and the bias is broken structurally. The judge is no longer a peer; it's a disinterested second opinion.

```bash
# Target: production prompt on gpt-4o.
# Judge: higher-tier disinterested model from a different vendor.
omegaprompt calibrate train.jsonl \
  --rubric rubric.json \
  --variants variants.json \
  --test test.jsonl \
  --target-provider openai \
  --target-model gpt-4o \
  --judge-provider anthropic \
  --judge-model claude-opus-4-7 \
  --output outcome.json
```

The `CalibrationArtifact` artifact records `target_provider`, `target_model`, `judge_provider`, `judge_model` — reproducibility is a machine-readable property of every run.

Another useful combination: local target via Ollama + cloud judge for grading.

```bash
# Target: local Llama (free inference). Judge: cloud-hosted for grading quality.
omegaprompt calibrate train.jsonl \
  --rubric rubric.json --variants variants.json --test test.jsonl \
  --target-provider openai \
  --target-base-url http://localhost:11434/v1 \
  --target-model llama3.1:70b \
  --judge-provider anthropic \
  --judge-model claude-opus-4-7
```

---

## The data contract

Every artifact this tool produces is Pydantic-validated. The data flows end-to-end:

```python
# Input: three user-authored files
Dataset(items=[DatasetItem(id="t1", input="...", reference="..."), ...])
JudgeRubric(
    dimensions=[
        Dimension(name="correctness", description="...", weight=0.5, scale=(1, 5)),
        Dimension(name="clarity",     description="...", weight=0.3),
        Dimension(name="conciseness", description="...", weight=0.2),
    ],
    hard_gates=[
        HardGate(name="no_refusal",        description="...", evaluator="judge"),
        HardGate(name="no_safety_violation", description="...", evaluator="judge"),
    ],
)
PromptVariants(
    system_prompts=["You are a...", "You are a senior...", "You are a teacher..."],
    few_shot_examples=[{"input": "...", "output": "..."}, ...],
)

# ↓ omega-lock.run_p1 drives PromptTarget.evaluate() for many (params, slice) combos
# Each evaluate() call issues N target calls + N judge calls for a given params dict.
# Target calls go through the target LLMProvider; judge calls go through a Judge
# (LLMJudge wraps a provider; RuleJudge is deterministic; EnsembleJudge combines).

# ↓ provider.call(ProviderRequest(response_schema_mode=STRICT_SCHEMA, output_schema=JudgeResult))
JudgeResult(
    scores={"correctness": 5, "clarity": 4, "conciseness": 3},  # integers within declared scale
    gate_results={"no_refusal": True, "no_safety_violation": True},
    notes="Response solves the task correctly. Slight padding in the wrap-up paragraph.",
)

# ↓ CompositeFitness aggregates (hard_gate × soft_score)
# fitness = sum(soft_score_i * all_gates_passed_i) / n_items

# ↓ omega-lock run_p1 emits walk-forward results
CalibrationArtifact(
    method="p1",
    unlock_k=3,
    best_params={"system_prompt_idx": 2, "few_shot_count": 1, "reasoning_profile": "deep"},
    best_fitness=0.876,
    walk_forward=WalkForwardResult(
        train_best_fitness=0.876,
        test_fitness=0.841,
        generalization_gap=0.040,
        kc4_correlation=0.84,
        passed=True,
    ),
    hard_gate_pass_rate=1.00,
    sensitivity_ranking=[{"axis": "system_prompt_idx", "gini_delta": 0.52, "rank": 0}, ...],
    status="OK",
    target_provider="openai", target_model="gpt-4o",
    judge_provider="anthropic", judge_model="claude-opus-4-7",
)
```

A malformed judge response raises `ValidationError` at the SDK boundary — it never pollutes the fitness. A missing hard-gate result in `JudgeResult.gate_results` is treated as "not answered," which is distinct from "passed." Every field in every model is a contract; violations are loud.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Dataset (.jsonl)   PromptVariants (.json)   JudgeRubric (.json) │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│  PromptTarget                                                 │
│    implements omega-lock CalibrableTarget                    │
│    param_space() ───▶ 6 meta-axes                             │
│    evaluate(params):                                          │
│       for each dataset item:                                  │
│         1. build ProviderRequest (system prompt + few-shot    │
│            + reasoning_profile + output_budget + schema_mode) │
│         2. target_provider.call(request) ─▶ ProviderResponse  │
│         3. judge.score(rubric, item, response) ─▶ JudgeResult │
│            (LLMJudge / RuleJudge / EnsembleJudge)             │
│       CompositeFitness(judge_results) ─▶ fitness              │
│    returns EvalResult(fitness, n_trials, metadata)            │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│  omega-lock run_p1                                            │
│    measure_stress + select_unlock_top_k                       │
│    GridSearch over unlocked subspace                          │
│    WalkForward on --test slice (KC-4 Pearson)                 │
│    emits grid_best, test_fitness, status                      │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│  CalibrationArtifact (JSON, schema_version=1.0)               │
│    best_params, best_fitness,                                 │
│    walk_forward{ train, test, gap, kc4, passed },             │
│    hard_gate_pass_rate, sensitivity_ranking,                  │
│    status (OK / FAIL_KC4_GATE / ...), rationale,              │
│    target/judge_provider+model, usage_summary                 │
└──────────────────────────────────────────────────────────────┘
```

Every piece above the omega-lock line is new; every piece at or below it is reused unchanged. The composition boundary is the `CalibrableTarget` protocol — one function (`param_space()`) plus one method (`evaluate(params)`).

This is why `PromptTarget` is ~200 lines. The expensive part (sensitivity analysis, search, walk-forward, kill criteria) already exists and is already tested by 176 unit tests in omega-lock. omegaprompt contributes the prompt-specific adapter, the LLM-as-judge machinery, and the fitness shape. Everything else is composition.

---

## Design decisions worth defending

**Provider-agnostic at two boundaries.** omegaprompt has exactly two places that touch a vendor SDK: the target call and the judge call. Both go through an `LLMProvider` Protocol. The calibration pipeline (stress, grid, walk-forward, KC-4) is vendor-neutral by construction. Target and judge accept *independent* providers — one line on the CLI flips to cross-vendor validation.

**Single-responsibility adapter over reimplementation.** omega-lock already handles stress measurement, top-K unlock, grid search, walk-forward, KC gates, benchmark scorecards, iterative lock-in, and `run_p2_tpe` for continuous-space search. Reimplementing any of that here would be wrong. The `PromptTarget` is a thin adapter; the value is that every omega-lock pipeline works on it unchanged regardless of which providers drive target and judge.

**Schema-enforced judge responses, per-vendor native.** Each adapter uses its vendor's strongest structured-output path — `messages.parse(output_format=JudgeResult)` on Anthropic, `beta.chat.completions.parse(response_format=JudgeResult)` on OpenAI. A malformed judge response — missing fields, scores out of declared scale, non-boolean gate outcomes — raises `ValidationError` at the SDK boundary. Without this, a single misbehaving judge call could tank an entire calibration run, and you would not notice until the final report. Regex-parsing a judge's prose output is the same mistake as trusting the model's self-reported line numbers in antemortem-cli: it works 99 times out of 100, and the 100th time poisons your entire calibration.

**Hard gates collapse fitness to zero, no gradient.** A soft penalty on refusal (e.g. "refusals lose 20%") rewards prompts that *almost* refuse — they capture most of the soft score while testing the safety boundary. Hard-zero punishes refusal absolutely. The searcher sees no reward signal from inside the refusal region, so it does not approach the boundary. This matches how real deployments evaluate prompts: a prompt that refuses 1-in-10 is not "90% as good," it is unshippable.

**Prompt caching on the judge system prompt, adapter-native.** The judge's ~5k-token system prompt is sized past each vendor's cacheable-prefix minimum. Anthropic uses explicit `cache_control={"type": "ephemeral"}`; OpenAI's automatic prompt caching fires once the system prompt clears the provider threshold. A typical calibration run issues hundreds of judge calls; cache hits dominate cost on every supported vendor. The CLI surfaces `cache_read_input_tokens` normalized across providers (OpenAI's `prompt_tokens_details.cached_tokens` maps to the same field), so silent invalidators fail loud regardless of where you're running. [The full judge prompt](src/omegaprompt/prompts.py) is worth reading as a case study in vendor-neutral prompt-cache-aware design.

**`reasoning_profile` and `output_budget` are first-class meta-axes, not globals.** Some prompts only need `light` reasoning; forcing `deep` across the board wastes tokens without improving fitness. Letting the calibration surface this is the entire point — if `reasoning_profile` has high stress on your dataset, it matters; if it has low stress, lock it at the neutral value and cut tokens. Same for `output_budget` (small / medium / large). Vendor translation (Anthropic `thinking` + `effort` vs OpenAI `reasoning_effort` vs system-prompt hint on plain chat) happens in each adapter — the axis semantics are provider-neutral.

**Target and judge providers are independent, first-class.** Passing the same provider to both is the common case. Splitting them unlocks (a) cross-vendor validation — an Anthropic judge grading OpenAI outputs (or vice-versa) breaks self-agreement bias structurally; (b) asymmetric quality/cost — cheap local target (Ollama) + cloud judge for grading; (c) testing — each side mocked independently; (d) billing isolation — route judge calls through a different workspace. The CLI exposes `--target-*` and `--judge-*` flags that mirror each other.

**Every path is normalized to forward slashes before going into a prompt.** `src\foo.py` and `src/foo.py` are different bytes in the API payload. Prompt caching is cache-invariant only if the bytes match. Windows users would silently lose cache hits otherwise — a ~\$15-cost-per-100-runs bug waiting to happen.

**No `temperature` / `top_p` axis.** Modern frontier-tier chat models increasingly remove or deprecate these (Anthropic's latest Claude pins drop them; OpenAI's o1/o3 reasoning models don't accept them). Rather than pretend to support them and fail on some servers, omegaprompt excludes them from the default meta-axes. If you need them for an older model, extend `MetaAxisSpace` with an extra axis and add the translation to your provider adapter — 20-line change, not a framework rewrite.

**UTF-8 with error='replace' in file reads.** Non-UTF-8 files (Windows files with BOM, legacy cp949 datasets, Latin-1 extraction prompts) do not crash the tool. They read with byte-level replacement and a CLI warning. This is the difference between "my calibration failed because of a BOM" and "my calibration ran and noted an encoding issue."

---

## What this is NOT

The discipline fails if you use it for the wrong thing. Explicit non-goals:

| This tool is NOT | Because |
|---|---|
| A prompt optimizer in the DSPy sense | DSPy synthesizes prompts via program abstraction + bootstrapped few-shot. omegaprompt *calibrates* prompts you've already written. If DSPy produces three candidate prompts, omegaprompt picks which one ships. They compose. |
| A prompt testing framework like promptfoo | promptfoo runs prompts against test cases with assertion-based grading. omegaprompt adds a *pre-declared walk-forward gate* on top. You can run promptfoo inside omegaprompt as one of the fitness dimensions. |
| A replacement for real evals on your actual traffic | Offline calibration on a curated dataset is the cheap screening step. Real-traffic A/B with business metrics is the ground truth. omegaprompt makes the offline step disciplined; it does not replace the online one. |
| A safety evaluator | `no_safety_violation` is a hard gate you can declare, but the judge is not a trained safety classifier. For serious safety evaluation, pair omegaprompt's calibration with a dedicated safety eval suite (e.g. AILuminate, HELM). |
| A benchmark for frontier LLM capability | The numbers are only as good as your judge rubric and your dataset. omegaprompt measures *your prompt on your data with your rubric*. Generalizing those numbers to other tasks is unsupported. |
| A free-money tool | Calibration burns API credits. A typical run is \$10–20. Budget accordingly; use cheaper models as judges during iteration, promote to stronger judges for the shipped run only. |

If you catch yourself using this tool for any of the above, you are using it wrong. The cost of wrong use is wasted API calls and, worse, false confidence in a prompt that still hasn't been evaluated against the thing that matters.

---

## Cost & performance

Per `evaluate()` call: 2 × (dataset_size) API calls — one target, one judge per item. A typical 10-item dataset = 20 API calls per candidate parameter set. Cost depends on which provider + tier you put on each boundary.

| Configuration (target / judge) | 10-item candidate | 125-candidate grid | With walk-forward |
|---|---|---|---|
| Anthropic frontier / Anthropic frontier | ~\$0.05–0.10 | ~\$6–12 | ~\$12–24 |
| OpenAI `gpt-4o` / OpenAI `gpt-4o` | ~\$0.03–0.06 | ~\$4–8 | ~\$8–16 |
| OpenAI `gpt-4o-mini` / Anthropic frontier | ~\$0.02–0.04 | ~\$2.5–5 | ~\$5–10 |
| Ollama local / OpenAI `gpt-4o` | ~\$0.015 (judge only) | ~\$2 (judge only) | ~\$4 (judge only) |
| Ollama local / Ollama local | free (compute only) | free | free |

Most cost-efficient iteration pattern: small local or mini-tier target while iterating prompts, frontier-tier judge to grade quality. Or run a frontier target and a cheaper judge during exploration; promote to frontier-both only for the final shipped calibration.

Every CLI invocation prints aggregate token usage at the end. `cache_read_input_tokens` is normalized across providers — the CLI reads Anthropic's native field and OpenAI's `prompt_tokens_details.cached_tokens` into the same slot. If it stays zero across consecutive runs, something in the judge prompt drifted — the CLI surfaces this explicitly rather than silently absorbing the cost. On local endpoints (Ollama), zero cache tokens is expected; ignore the warning.

---

## Validation

**110 tests, 0 network calls in CI.** Every provider (current and future) is accepted via the `LLMProvider` Protocol, so every API test mocks with `SimpleNamespace` or `MagicMock`. The test surface asserts the *exact* shape of each request payload (model, `response_format`, thinking config, cache_control placement, few-shot ordering) without negotiating with a real server.

| Module | Coverage |
|---|---|
| `domain/` | `PromptVariants` / `MetaAxisSpace` / `CalibrationArtifact` / `Dataset` / enums — required fields, range validation, JSON roundtrip, provider metadata fields, ordinal clamping. |
| `judges/` | `RuleJudge` (no_refusal / non_empty / json_object / regex / missing-check raises), `LLMJudge` (STRICT_SCHEMA dispatch, payload contents, non-JudgeResult error), `EnsembleJudge` (rule-first short-circuit, LLM escalation, merged gate_results). |
| `core/` | `CompositeFitness` (empty / all-pass / partial / all-fail / per-item preserved), `measure_sensitivity` (Gini ordering, top-K unlock), `evaluate_walk_forward` (gap math, KC-4 Pearson, zero-variance skip), `save_artifact` / `load_artifact` roundtrip. |
| `providers/` | Factory rejects unknown names, respects `base_url`. Anthropic adapter translates meta-axes (DEEP → thinking + effort=high; JSON_OBJECT → system-prompt suffix; STRICT_SCHEMA → messages.parse) and raises on refusal. OpenAI adapter translates meta-axes (reasoning_effort, response_format, beta parse path), normalizes `prompt_tokens_details.cached_tokens` → `cache_read_input_tokens`, raises on content_filter. |
| `api.py` | Legacy v0.2 shim — `call_target` translates old kwargs (effort / max_tokens / thinking_enabled) into a v1.0 `ProviderRequest`. |
| `targets/` | `PromptTarget` — end-to-end with mocked providers + judges, meta-axis param resolution and clamping, usage accumulation. |
| `cli.py` | Help / version / subcommand wiring + `report` markdown rendering + `diff` regression detection. |

Run with `uv run pytest -q`. Typical wall time: under one second.

---

## The 3-layer stack

omegaprompt is the applied layer in a three-project system:

```
       ┌─────────────────────────────────────────────┐
LAYER  │  omegaprompt  (this repo)                   │  "Apply the discipline to prompts"
APPLY  │  v1.0.0 — model-agnostic prompt calibration │
       └────────────────────┬────────────────────────┘
                            │ depends on
                            ▼
       ┌─────────────────────────────────────────────┐
LAYER  │  omega-lock                                 │  "The calibration framework"
CORE   │  v0.1.4 — stress + grid + walk-forward + KC │
       └────────────────────┬────────────────────────┘
                            │ validated by
                            ▼
       ┌─────────────────────────────────────────────┐
LAYER  │  Antemortem + antemortem-cli                │  "The discipline around the build"
META   │  methodology + tooling for pre-impl recon   │
       └─────────────────────────────────────────────┘
```

- **[omega-lock](https://github.com/hibou04-ops/omega-lock)** — supplies the calibration engine: stress measurement, top-K unlock, grid search, walk-forward, kill criteria, benchmark scorecards. 176 tests. Shipped 2026-04-18.
- **[Antemortem](https://github.com/hibou04-ops/Antemortem)** + **[antemortem-cli](https://github.com/hibou04-ops/antemortem-cli)** — the pre-implementation reconnaissance discipline under which both omega-lock and omegaprompt were built. Antemortem catches ghost traps before code is written; omega-lock catches overfit parameters before they ship; omegaprompt catches overfit prompts before they deploy. The pattern repeats at three scales: spec, parameters, prompts.

The layering matters for credibility. The calibration engine was shipped and validated before this prompt adapter was written. omegaprompt is ~200 lines of adapter code because everything it needs already exists.

---

## Relation to adjacent tools

| Tool | What it does | What omegaprompt adds |
|---|---|---|
| **[promptfoo](https://www.promptfoo.dev/)** | Run prompts against test cases; assertion-based grading. | Pre-declared walk-forward gate (KC-4) so training ≠ ship criterion. Hard gates that collapse fitness to zero, not softly penalize. Stress-based axis selection. Composable — promptfoo output can be one of several judges. |
| **[DSPy](https://dspy.ai/)** | Prompt synthesis via program abstraction + bootstrapped few-shot. | Domain-agnostic adapter (any `CalibrableTarget` works). Calibration-first framing (stress + grid + walk-forward), not program synthesis. DSPy output is just another `system_prompt_variant` in the search space. |
| **Optuna / Ray Tune** (applied to prompts) | General HPO over prompt knobs. | Walk-forward validation + pre-declared kill criteria out of the box. Schema-enforced LLM-as-judge responses. Composite `hard_gate × soft_score` fitness built in, not reinvented per project. |
| **Custom "eval suites"** | Project-specific scripts that call the model, score, rank. | Structured data contract (`Dataset`, `Rubric`, `Outcome`), machine-readable artifact, reproducibility, plug-and-play into a calibration engine that ships with a 30-run reference benchmark. |
| **Anthropic's built-in [evals](https://docs.anthropic.com/claude/docs/evaluate-and-improve-performance-of-claude-models)** | Provider-native eval workflow, `messages.create` + rubric. | The same infrastructure plus discipline: you cannot declare the threshold after you've seen the results. The Antemortem/omega-lock lineage forces pre-declaration. |

The USP is *discipline, not search.* The search part is handled by omega-lock (which handles it for any `CalibrableTarget`). omegaprompt contributes the prompt-specific adapter and the hard-gates-first fitness shape.

---

## FAQ for skeptics

**Isn't this just promptfoo + a walk-forward step?**

Kind of — and that is the point. The walk-forward step *is* the entire discipline. promptfoo runs the examples; omegaprompt makes the ranking mechanically falsifiable. Without KC-4, promptfoo gives you numbers that may or may not mean anything. With KC-4, a failing run tells you to go back and iterate before you ship.

**Why not use `temperature` as an axis?**

Modern frontier models increasingly remove the sampling knobs (Anthropic's latest Claude pins; OpenAI's o1/o3 reasoning family). The contract is *"we pick the sampling for you; you pick reasoning profile and prompt shape."* Calibrating a removed knob is nonsense. For older models that still accept `temperature`, extend `MetaAxisSpace` with a `temperature_idx` axis and add the translation to your provider adapter — 20-line change, not a framework rewrite.

**Won't the judge just agree with the target if both are the same model?**

Not for well-written rubrics. The judge is prompted to *evaluate against a specific rubric* with *specific integer scales* and *binary gates*. The two calls are independent — different system prompts, different task framings. The judge has no memory of the target call. That said, using a *different* model as judge (ideally stronger) is the recommended production setup and the v0.2 `multi-judge` pattern will formalize it.

**What if my judge is miscalibrated?**

Then your calibration is miscalibrated. Judge quality is a first-class concern. Three defenses: (1) the judge's ~5k-token system prompt includes explicit calibration-hygiene instructions ("anchor on midpoint, use full range, agree with yourself"); (2) the rubric schema requires per-dimension integer scales and clamps out-of-range scores; (3) the v0.2 `multi-judge` pattern compares `judge_v1` vs `judge_v2` rankings — disagreement is a trust signal, not a result.

**How is this different from Anthropic's built-in evals?**

Anthropic's evals are the native API surface. omegaprompt adds *pre-declared* gating on top: the rubric is fixed before the run, the threshold is in config, and you cannot adjust either after you've seen the scores. This is the Winchester defense, borrowed from quant-finance: kill criteria declared up-front cannot be relaxed. Native evals let you do this manually; omegaprompt enforces it structurally.

**Does this work on non-Anthropic models?**

Yes. v1.0 is provider-neutral at both the target and judge call boundaries. Anthropic, OpenAI, and any OpenAI-compatible endpoint (Azure, Groq, Together.ai, OpenRouter, local Ollama) are supported out of the box; adding another vendor is one `LLMProvider` implementation plus a registry line. Meta-axes (`reasoning_profile`, `output_budget`, `response_schema_mode`, `tool_policy`) map to whatever each vendor calls them internally. Cross-vendor validation — target on one vendor, judge on another — breaks self-agreement bias structurally and is a one-flag CLI operation (`--target-provider openai --judge-provider anthropic`).

**Can I run this on private/proprietary datasets?**

Yes. Dataset + rubric + variants are all local JSON/JSONL. The tool reads them locally, sends them to whichever API you configured for target / judge, and writes the artifact back to disk. If you point both providers at local Ollama (`--target-base-url http://localhost:11434/v1 --judge-base-url http://localhost:11434/v1`), the data never leaves your laptop.

**What about cost?**

A calibration run is \$10–20 at current frontier pricing for a 10-item dataset, 125-candidate grid, walk-forward. Two mitigations: use a cheaper model as judge during iteration (4–5× reduction), and cache-aware prompts dominate cost on the judge side so the *second* run in a 5-minute window is ~50% the cost of the first. Budget accordingly.

**How do I know if my calibration actually generalized?**

Read `generalization_gap` on the outcome. Under 10% = strong generalization. 10–25% = acceptable for non-critical paths. Above 25% = your training slice does not represent production; expand the dataset. The `status` field says `PASS` or `FAIL:KC-4` — if it says FAIL, do not ship regardless of the scores.

---

## Prior art & credit

The three ideas this tool stands on:

- **Train/test split with a pre-declared gate** — the foundational ML defense, documented in every undergraduate ML curriculum. The specific form used here (Pearson rank correlation as the gate, KC-4) comes from [omega-lock](https://github.com/hibou04-ops/omega-lock)'s kill-criteria framework.
- **LLM-as-judge** — pattern popularized in *[Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)* (Zheng et al., 2023). omegaprompt implements the pattern with schema enforcement (Pydantic + `messages.parse`) to defend against the known failure mode of free-form judge responses contaminating the scoring pipeline.
- **The Winchester defense** — originally a quant-finance discipline: *kill criteria declared before the run cannot be relaxed after.* Used here to argue that `KC-4` must be enforced in config, not adjusted post-hoc on inspection of scores.

The naming: *omega-lock* (parameter calibration) → *omegaprompt* (prompt calibration) is the intentional family branding. omega-lock was extracted from a trading-strategy calibration experiment that ended in `KC-4 FAIL` — the overfitting defense firing exactly as designed. omegaprompt is the same insight applied one layer up.

---

## Status & roadmap

v1.0.0 is the first stable release of the provider-neutral redesign. The data contract (`Dataset`, `JudgeRubric`, `PromptVariants`, `MetaAxisSpace`, `CalibrationArtifact`) is stable. The CLI contract (`omegaprompt calibrate` / `report` / `diff`, their flags, exit codes) is stable. The judge prompt iterates in-place — expect v1.0.x bumps for judge prompt revisions tracked in CHANGELOG under *"Judge prompt revisions."*

Semver applies strictly from v1.0 onward.

**v0.1.x (judge prompt iteration track)**
- Dogfood against diverse task types (code generation, reasoning, extraction, classification). Record scoring drift.
- Reference scoring-quality benchmark so judge prompt revisions are measured, not guessed.
- Additional hard-gate evaluators (format predicates, safety classifiers) callable without a judge round-trip.

**v1.1 (tooling depth)**
- Multi-judge validation pattern: `judge_v1` + `judge_v2` over top-K, disagreement = trust signal.
- `--dry-run` with cost estimate before launching a calibration run.
- Additional provider adapters: Gemini, native HuggingFace Inference, local llama.cpp.

**v1.2 (ecosystem)**
- Benchmark harness: multiple (task × rubric × seed) combinations, RAGAS-style scorecard like omega-lock's.
- GitHub Action for CI gating — runs a calibration on PR, blocks merge on regression detected by `omegaprompt diff`.
- HTML report rendering (`omegaprompt report --format html`) for PR status checks.

**Explicitly out of scope:** web dashboard, proprietary hosting, multi-user tenancy. omegaprompt is a local developer tool; keep it local.

Full changelog: [CHANGELOG.md](CHANGELOG.md).

---

## Contributing

The most valuable contributions are published calibration artifacts — a dataset, a rubric, and the resulting `CalibrationArtifact.json` across methods. They make the judge prompt evidence-based.

Issues and PRs welcome. For non-trivial changes, run an antemortem first with [`antemortem-cli`](https://github.com/hibou04-ops/antemortem-cli) — we dogfood the discipline that built this framework.

---

## Citing

```
omegaprompt v1.0.0 — model-agnostic calibration engine for LLM prompts.
https://github.com/hibou04-ops/omegaprompt, 2026.
```

Parent framework:
```
omega-lock v0.1.4 — sensitivity-driven coordinate descent calibration framework.
https://github.com/hibou04-ops/omega-lock, 2026.
```

Methodology (how this and its siblings were built):
```
Antemortem v0.1.1 — AI-assisted pre-implementation reconnaissance for software changes.
https://github.com/hibou04-ops/Antemortem, 2026.
```

---

## License

MIT. See [LICENSE](LICENSE).

## Colophon

Designed, implemented, and shipped solo. Adapter layer over omega-lock; zero calibration-engine reimplementation. 110 tests, 0 live API calls in CI. The tool is built with the same pre-implementation reconnaissance discipline it supports for its callers.
