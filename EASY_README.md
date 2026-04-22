# omegaprompt — Easy Start

> The short version, for people who found the 16-section academic README intimidating.
> Full doc: [README.md](README.md) · 한국어 Easy: [EASY_README_KR.md](EASY_README_KR.md)

## What problem does it fix?

You iterate prompt variants against 20 hand-picked examples. The top scorer gets shipped. **On day two in production, it fails on inputs the 20 examples didn't represent.**

That's overfitting. ML has known the defense since the 1990s: a **held-out test slice**, a **pre-declared correlation threshold**, and a **gate that blocks ship if the numbers miss**. Every ML textbook teaches it. Most prompt-tuning tools skip it.

omegaprompt is prompt calibration with those three defenses wired in. It's **provider-neutral** (same artifact replays across Anthropic / OpenAI / local / OpenAI-compatible) and it **records every degradation** so CI sees when a provider silently dropped a capability.

## 60-second mental model

```
Your dataset  →  search over meta-axes  →  walk-forward on held-out  →  PASS or FAIL_KC4_GATE
                 (reasoning, budget,        (Pearson + gap gate)        ship-ready artifact
                  schema, variants)
```

You define: **dataset + rubric + candidate prompts + provider**.
It returns: **a `CalibrationArtifact` JSON** with neutral-baseline vs calibrated fitness, uplift, walk-forward metrics, capability degradation events, and a ship recommendation.

## Install

```bash
pip install omegaprompt
```

## The easiest path: the CLI

```bash
export ANTHROPIC_API_KEY=sk-ant-...

omegaprompt calibrate train.jsonl \
  --test test.jsonl \
  --rubric rubric.json \
  --variants variants.json \
  --output result.json \
  --target-provider anthropic \
  --judge-provider anthropic \
  --profile guarded
```

You get `result.json` — the `CalibrationArtifact`. Look at `.status` (`OK` / `FAIL_KC4_GATE` / `FAIL_HARD_GATES`) and `.ship_recommendation` (`SHIP` / `HOLD` / `EXPERIMENT` / `BLOCK`).

## The Python path

```python
from omegaprompt import (
    Dataset, Dimension, HardGate, JudgeRubric,
    PromptVariants, PromptTarget, LLMJudge, make_provider,
)
from omega_lock import run_p1

# 1. Dataset + rubric
train = Dataset.from_jsonl("train.jsonl")
test  = Dataset.from_jsonl("test.jsonl")

rubric = JudgeRubric(
    dimensions=[Dimension(name="accuracy", description="is it correct?", weight=1.0)],
    hard_gates=[HardGate(name="no_refusal", description="model must try", evaluator="judge")],
)

# 2. Candidate prompts (at least 2 for sensitivity signal)
variants = PromptVariants(
    system_prompts=["You are a helpful assistant.", "Be terse. Be accurate."],
    few_shot_examples=[],
)

# 3. Provider + judge
provider = make_provider("anthropic")           # or "openai", "local", "ollama", "vllm"
judge    = LLMJudge(provider=provider)

# 4. Wrap as calibrable targets
train_t = PromptTarget(target_provider=provider, judge=judge, dataset=train, rubric=rubric, variants=variants)
test_t  = PromptTarget(target_provider=provider, judge=judge, dataset=test,  rubric=rubric, variants=variants)

# 5. Run — uses omega-lock's run_p1 under the hood
result = run_p1(train_target=train_t, test_target=test_t)

# Best prompt is in result.grid_best["unlocked"]
```

## The 6 meta-axes (the actual search space)

Instead of vendor-specific parameter names, omegaprompt searches over semantic categories each provider adapter translates internally:

| Axis | Values | What it controls |
|---|---|---|
| `system_prompt_variant` | index into your prompts | Which system prompt |
| `few_shot_count` | 0, 1, 2, ... | How many examples to include |
| `reasoning_profile` | OFF / LIGHT / STANDARD / DEEP | Extended thinking depth |
| `output_budget_bucket` | SMALL (1024) / MEDIUM (4096) / LARGE (16000) | max_tokens |
| `response_schema_mode` | FREEFORM / JSON_OBJECT / STRICT_SCHEMA | Output structure enforcement |
| `tool_policy_variant` | NO_TOOLS / TOOL_OPTIONAL / TOOL_REQUIRED | Tool-use policy (declared; not wired in providers yet) |

Same artifact replays across vendors because the axes are semantic.

## Three judges

- **`RuleJudge`** — deterministic regex/predicate gates (no_refusal, JSON-valid, regex-match). Zero API cost.
- **`LLMJudge`** — a capable LLM scores dimensions via STRICT_SCHEMA. Ships a judge system prompt that actively resists self-congratulation.
- **`EnsembleJudge`** — RuleJudge first; if any rule gate fails, **short-circuit** (no LLM call). Otherwise escalate. Saves cost on broken responses.

## The two profiles

| | `guarded` (default) | `expedition` |
|---|---|---|
| Silent schema fallback | ❌ raise | ✅ allow, log as CapabilityEvent |
| Placeholder providers | ❌ raise | ✅ allow |
| Non-ship-grade judge | ❌ raise | ✅ allow |
| Default `--max-gap` | 0.25 | 0.35 |
| Default `--min-kc4` | 0.5 | 0.3 |

Every relaxation in `expedition` is recorded as a `RelaxedSafeguard` entry in the artifact. The bargain between strictness and reach is explicit and auditable.

## When it's worth it

- You have a **real train/test split** (or can make one).
- Someone downstream needs to **trust** the prompt — ops, compliance, you in 3 months.
- You want to **replay the same calibration on a different vendor** without rewriting search axes.
- You need to **catch silent provider degradations** (e.g., a local endpoint dropping strict schema).

## When it's overkill

- One-off prompt for a demo.
- No test set, no holdout, nobody will review the result.
- You're fine eyeballing 10 outputs and shipping.

In those cases, just iterate in a playground.

## Optional preflight plugins

`omegaprompt` ships the **plugin interface** (`omegaprompt.preflight.contracts` + `.adaptation`) but no probes. Two separate packages fill the slot:

- **[mini-omega-lock](https://pypi.org/project/mini-omega-lock/)** — measures *your actual environment* (judge consistency, endpoint reliability, context margin, latency). Live API calls.
- **[mini-antemortem-cli](https://pypi.org/project/mini-antemortem-cli/)** — classifies *your config* against 7 calibration trap patterns (self-agreement, small-sample power, variant homogeneity, ...). Pure deterministic rules, zero API calls.

Install only if you want adaptive thresholds tuned to your real setup.

## Go deeper

- Full academic README: [README.md](README.md) (system architecture, data contracts, all the appendices)
- Meta-axes definitions: `src/omegaprompt/domain/enums.py`
- Fitness rule (hard_gate × soft_score): `src/omegaprompt/core/fitness.py`
- Walk-forward gate: `src/omegaprompt/core/walkforward.py`
- Provider adapters: `src/omegaprompt/providers/`

License: Apache 2.0. Copyright (c) 2026 hibou.
