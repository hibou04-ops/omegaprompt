# omegaprompt

**The overfit gate for your prompts.** Your prompt aced your eval set — that's exactly why you can't trust it yet. `omegaprompt` re-tests the winning prompt on examples it never tuned on, and **fails your CI build** if it doesn't hold up.

[![CI](https://github.com/hibou04-ops/omegaprompt/actions/workflows/ci.yml/badge.svg)](https://github.com/hibou04-ops/omegaprompt/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/pypi/l/omegaprompt?color=blue&label=license&cacheSeconds=3600)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/omegaprompt?color=blue&cacheSeconds=3600)](https://pypi.org/project/omegaprompt/)
[![PyPI](https://img.shields.io/pypi/v/omegaprompt?color=blue&label=pypi&cacheSeconds=3600)](https://pypi.org/project/omegaprompt/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Artifact schema](https://img.shields.io/badge/artifact-schema%20v2.0-blueviolet.svg)](#8-the-calibrationartifact-schema-v20)
[![MCP](https://img.shields.io/badge/MCP-server-blueviolet.svg)](#103-mcp-server-claude-code-cursor)
[![Parent framework](https://img.shields.io/badge/framework-omega--lock-blueviolet.svg)](https://github.com/hibou04-ops/omega-lock)

Docs: **[Easy start](https://github.com/hibou04-ops/omegaprompt/blob/main/EASY_README.md)** · [Full reference](https://github.com/hibou04-ops/omegaprompt/blob/main/README.md) · [한국어](https://github.com/hibou04-ops/omegaprompt/blob/main/README_KR.md) · [쉬운 한국어](https://github.com/hibou04-ops/omegaprompt/blob/main/EASY_README_KR.md) · [Examples gallery](examples/) · [Claim ledger (trust evidence)](docs/claims/README_CLAIMS.generated.md)

Keywords: **prompt overfitting · prompt regression testing · LLM eval CI · prompt evaluation · prompt A/B test in CI · held-out validation for prompts · CI ship gate for prompts**

```bash
pip install omegaprompt              # core
pip install "omegaprompt[mcp]"       # + MCP server (Claude Code / Cursor)
```

> **v2.1.0 (2026-06-12)** — new `omegaprompt gate` CI ship-gate command (one exit code: ship or block on held-out transfer + overfit gap), `--format json` on `report`/`diff` for stable CI summaries, `--format html` single-file scorecard, a keyless **`ollama`** local provider, a GitHub composite **Action** (`uses: hibou04-ops/omegaprompt@v2.1.0`), and a machine-readable **overfit-metrics** block (`extract_overfit_metrics`). MCP tool set is unchanged (frozen at 8 — no `gate` MCP tool); artifact schema stays `2.0`; backward compatible. Exact deterministic reference metrics are tracked in the generated [claim ledger](docs/claims/README_CLAIMS.generated.md).

<!-- public-claim-ledger:start -->
> Claim evidence source: [docs/claims/public_claim_ledger.json](docs/claims/public_claim_ledger.json), rendered by `python tools/generate_readme_claims.py`.
<!-- public-claim-ledger:end -->

Trust docs: [trust model](docs/trust-model.md) · [toolkit positioning](docs/toolkit-positioning.md) · [provider capabilities](docs/provider-capabilities.md) · [profiles and risk boundaries](docs/profiles-and-risk-boundaries.md) · [release checklist](docs/release/release-checklist.md) · [post-release verification](docs/release/release-checklist.md#post-release-network-verification).

---

## Your prompt is overfit to your eval set — and you don't know it

You tune a handful of prompt variants against your 30-example eval set. Variant #5 wins — 4.8/5. You ship it.

A week later, production quality is *worse* than before. Why?

Because 4.8 was the score on **the exact examples you tuned against**. You didn't measure a prompt — you memorized an answer key. That's overfitting, and your eval tool reported PASS because that's all it was asked to do. ML solved this in the 1990s with a train/test split. Most prompt workflows still ship without one.

`omegaprompt` is the 60-second check that catches this **before** you ship:

1. It tunes your prompt on a **train** slice (across system prompt, few-shot, reasoning effort, output budget, response-schema mode, tool policy).
2. It **re-tests the winner on a held-out slice it never tuned on**.
3. It **ships only if held-out performance tracks train performance** — two thresholds you declare *before* scoring, so nobody quietly lowers the bar to make a prompt pass.

Then one line in CI makes a "small prompt tweak" fail the build if it silently degrades quality.

> **Your eval says PASS. omegaprompt says it won't generalize.** That sentence is the whole product.

---

## It sits on top of your eval — it doesn't replace it

omegaprompt does **not** compete with promptfoo, DSPy, deepeval, Braintrust, or your hand-rolled harness. Those tools *find and score* the best prompt. omegaprompt does the one thing they leave out: a **train/held-out split with a transfer gate** that tells you whether that best prompt survives data it wasn't tuned on — plus a **ship/block CI verdict** so the next PR's "tiny wording tweak" can't silently tank production.

| | promptfoo / DSPy / your harness | Just eyeball the eval | **omegaprompt** |
|---|---|---|---|
| Run prompts against test cases | ✓ | manual | reuses yours as input |
| Find / optimize the best prompt | ✓ (that's their job) | by hand | not its job |
| Train / held-out split | ✗ (one set, scored once) | ✗ | ✓ pre-declared; tuner never sees held-out |
| Held-out transfer gate (does train score predict held-out score?) | ✗ | ✗ | ✓ per-item correlation gate |
| Max train↔held-out gap gate | ✗ | ✗ | ✓ pre-declared threshold |
| Single-command CI ship/block verdict | partial | ✗ | ✓ `gate` / `diff` exit non-zero |
| Machine-readable "is it overfit?" number | ✗ | ✗ | ✓ `extract_overfit_metrics` |
| Overfit caught **before** ship | ✗ | ✗ | ✓ that's the entire point |

> **One line for your tech lead:** promptfoo/DSPy tell you *which* prompt scored best. omegaprompt tells you *whether that prompt holds up* on held-out data — and gives CI a single exit code to ship or block on it.

Your existing eval outputs plug straight in — assertions become rule-based gates, your dataset becomes the train/held-out source. omegaprompt is **audit-first, not search-first**: it assumes you already picked candidates and answers the question downstream of search — *did you actually generalize?*

---

## 30-second demo — no API keys, no network

https://github.com/user-attachments/assets/d4308cc3-b8c1-4bb7-b67d-f763e6c26f11

The fastest way to watch the gate run is the deterministic offline replay. It uses built-in in-memory model + judge stand-ins, so it needs **no provider keys and makes no network calls** — every run is byte-identical:

```bash
git clone https://github.com/hibou04-ops/omegaprompt.git
cd omegaprompt && pip install -e .

# Replay the deterministic offline calibration (no keys, no network)
PYTHONIOENCODING=utf-8 python examples/demo_replay.py
```

You'll see the actual gate output:

```
status: OK
ship_recommendation: ship

neutral_fitness:    0.4250     # baseline prompt, no tuning
calibrated_fitness: 0.9250     # winner on the TRAIN slice
uplift_percent:     117.65%    # how much tuning helped on TRAIN
test_fitness:       0.9250     # SAME winner on the HELD-OUT slice
generalization_gap: 0.00%      # train vs held-out — small gap = it transferred
kc4_status:         MISSING_PER_ITEM_SCORES
```

**Honest read of this demo:** the bundled demo dataset has *disjoint* train/held-out items (no shared item ids), so the per-item transfer gate cannot fire — it reports `MISSING_PER_ITEM_SCORES` and the gate degenerates to the gap check alone (which here is 0.00%). The per-item transfer gate is what *actually* fires on a real **paired** dataset (train and held-out sharing item ids). So don't read this demo's clean numbers as "the transfer gate passed" — read it as "the gap check passed and the transfer gate had nothing to score." See *How it works*.

Turn that same artifact into a one-command CI verdict:

```bash
omegaprompt gate examples/reference/reference_artifact.json
# exit 0 = clear to ship · 1 = ship-blocked (overfit / unmeasured) · 2 = environment/load error

omegaprompt gate examples/reference/reference_artifact.json --format json   # machine summary for CI
```

Inspect any shipped artifact's integrity offline, anytime:

```bash
omegaprompt check-artifact examples/reference/reference_artifact.json --strict
```

---

## Run it for real — your dataset, your provider

```bash
export ANTHROPIC_API_KEY=...      # or OPENAI_API_KEY / GEMINI_API_KEY — or run keyless against Ollama

omegaprompt calibrate train.jsonl \
  --test test.jsonl \                    # held-out slice the winner is re-tested on
  --rubric rubric.json \                 # your judging rubric
  --variants variants.json \             # candidate system prompts + few-shot
  --target-provider anthropic \
  --judge-provider openai \              # cross-vendor judge avoids self-grading bias
  --min-kc4 0.5 \                        # held-out transfer gate, declared up front
  --max-gap 0.25 \                       # max allowed train↔held-out gap
  --output artifact.json
```

`--min-kc4` is the **held-out transfer gate** and `--max-gap` is the **max train↔held-out gap** — both are thresholds you set *before* scoring. Output is a single JSON `CalibrationArtifact` with a verdict: `.status` is `OK` / `FAIL_KC4_GATE` / `FAIL_HARD_GATES`, and `.ship_recommendation` is `ship` / `hold` / `experiment` / `block`. `calibrate` **exits non-zero** when the verdict isn't ship-clean, so it gates straight from the command line. Render it for a PR or review:

```bash
omegaprompt report artifact.json > report.md            # Markdown (default)
omegaprompt report artifact.json --format html > report.html   # self-contained scorecard, no JS
omegaprompt report artifact.json --format json          # stable, schema-versioned CI summary
```

---

## Drop it into CI — a prompt A/B test that fails the build

A prompt change is a code change. Gate it like one. The dedicated **`omegaprompt gate`** command is the CI hero: it fuses an offline integrity audit with the held-out transfer/gap (overfit) verdict and **exits `0` (ship) / `1` (block) / `2` (environment)**. The repo ships a GitHub composite Action so it's one `uses:` line:

```yaml
# .github/workflows/prompt-audit.yml
- uses: hibou04-ops/omegaprompt@v2.1.0
  with:
    artifact: artifact.json          # a CalibrationArtifact you produced in a prior step
    format: json                     # machine-readable gate summary
    require-generalization: "true"   # an absent/unverifiable transfer verdict blocks the build
```

Prefer a raw step? It's the same command:

```yaml
- run: pip install omegaprompt
- run: |
    omegaprompt calibrate train.jsonl --test test.jsonl \
      --rubric rubric.json --variants variants.json \
      --target-provider anthropic --output new.json
- run: omegaprompt gate new.json --format json    # exit 1 on overfit/unverified — fails the build
- run: omegaprompt diff baseline.json new.json    # exit 1 on regression vs a known-good baseline
```

`gate` answers "is *this* artifact clear to ship?"; `diff` answers "did this PR regress against last time?" Now "I just tweaked the system prompt" is a reviewable, gated change — not a roll of the dice. A complete example workflow lives at [`examples/ci/ship-gate.yml`](examples/ci/ship-gate.yml).

---

## The five commands

| Command | What it does |
|---|---|
| `omegaprompt calibrate` | Tune on train, re-test the winner on held-out, write the artifact + ship verdict. Non-zero exit on gate failure. |
| `omegaprompt report` | Render an artifact as Markdown, a single-file HTML scorecard (`--format html`), or a stable CI JSON summary (`--format json`). |
| `omegaprompt diff` | CI regression gate between two artifacts. **Non-zero exit on regression.** `--format json` for a deterministic machine diff. |
| `omegaprompt check-artifact` | Zero-network integrity check before you trust an artifact as ship evidence (`--strict` for CI). |
| `omegaprompt gate` | **CI ship gate** (new in 2.1.0): integrity + held-out transfer/gap verdict in one command, exit `0/1/2`, `--format json`. The thing you actually wire into CI. |

Also installed: `omegaprompt-mcp` (the agent server launcher) and `omegacal` (a compatibility alias for the same CLI).

---

## "Is my prompt overfit?" — one machine-readable number

The two numbers that answer this — the **transfer correlation** (per-item train↔held-out agreement) and the **train↔held-out gap** — are surfaced as one prominent block you can read from code or CI:

```python
from omegaprompt import extract_overfit_metrics
import json

artifact = json.load(open("artifact.json"))
m = extract_overfit_metrics(artifact)
print(m.overfit_verdict)        # GENERALIZES / OVERFIT / UNVERIFIABLE / UNKNOWN
print(m.transfer_correlation)   # per-item r, or None when the split is disjoint
print(m.generalization_gap)     # train fitness minus held-out fitness
```

The same block is embedded in `omegaprompt report --format json` and `omegaprompt gate --format json`, so a CI step (or a coding agent) can read the verdict without parsing prose. This is a **pure read** over the existing artifact — it adds no field, so the artifact schema stays `2.0` and every golden hash is byte-stable.

---

## How it works

You give it a **dataset + a judging rubric + candidate prompts**. It searches structured, provider-neutral variation axes and scores candidates on the train slice:

| Axis | What it varies |
|---|---|
| System-prompt variant | which of your system prompts |
| Few-shot count | how many examples to include |
| Reasoning effort | off / light / standard / deep |
| Output budget | small / medium / large token ceiling |
| Response-schema mode | freeform / JSON object / strict schema |
| Tool policy | no tools / optional / required *(declared; not yet wired in providers)* |

Then it re-tests the winning configuration on the **held-out** slice and applies two pre-declared gates:

- **Held-out transfer gate** (`--min-kc4`): per-item held-out scores must still track the target. If train looks great but the held-out items stop correlating, the prompt overfit — the run is flagged `FAIL_KC4_GATE`. *(This needs train and held-out to share item ids — a "paired" replay. On an ordinary disjoint split it degrades to the gap check below, which still catches overfit — that's exactly what the offline demo above shows.)*
- **Max gap gate** (`--max-gap`): the train↔held-out fitness gap must stay under your declared ceiling.

Both thresholds are set **before** scoring. There's no lowering the bar after you see results.

> **On terminology:** this is **held-out validation** — a held-out test slice, not time-series forecasting. Prompts aren't a time series, so there's no real "future" to walk into; it's a holdout split. (The underlying calibration engine still names the routine `walk-forward` internally; in plain terms, read it as *holdout*.)

### Two modes

- **Strict mode** (default) — silent schema fallbacks, placeholder providers, or non-ship-grade judges **raise** instead of passing quietly, and the held-out gates use tight defaults. Use this for anything you'll ship.
- **Fast mode** — allows those relaxations for quick local exploration, but **records every one** on the artifact so the looser run stays auditable. Looser default gates.

### Providers and agents

- **Provider adapters:** Anthropic, OpenAI, Gemini, a generic `local` OpenAI-compatible adapter, and a dedicated keyless **`ollama`** adapter (local `http://localhost:11434/v1`). Other OpenAI-compatible backends (vLLM, llama.cpp, and any `base_url`: Azure, Groq, Together, OpenRouter) are reached through the `local` adapter — so the full provider set is `anthropic / openai / gemini / local / ollama / vllm / llama_cpp`. The same artifact replays across vendors because the axes are semantic, not vendor-specific knobs. Use a **cross-vendor judge** (e.g. Anthropic target, OpenAI judge) so the grader isn't a peer of the thing it's grading.
- **MCP server (8 tools):** `calibrate`, `evaluate`, `report`, `diff`, `measure_sensitivity`, `grade`, `preflight`, `classify_traps` — so a coding agent can run the gate and read the verdict before opening a PR. *(The MCP tool set is frozen at 8; the new `gate` command is CLI- and Python-only, not an MCP tool.)*

```bash
pip install "omegaprompt[mcp]"
python -m omegaprompt.mcp           # stdio; Claude Code / Cursor spawn it as a subprocess
```

```json
{ "mcpServers": { "omegaprompt": { "command": "python", "args": ["-m", "omegaprompt.mcp"] } } }
```

---

## When to use it

**Worth it:** you have (or can make) a real train/held-out split; someone downstream has to trust the prompt — ops, compliance, future-you; you want to replay the same calibration on another vendor; you want a prompt PR to fail CI on regression.

**Overkill:** a one-off demo prompt; no held-out set and nobody reviewing the result; you're fine eyeballing 10 outputs. Then just iterate in a playground — this tool buys you nothing.

**Honest scope:** offline held-out is a cheap *screen*, not production ground truth. The numbers reflect *your rubric on your dataset on your models* — not a vendor benchmark. It's not a safety classifier and not a substitute for production A/B telemetry. See [Limitations](#14-limitations-and-scope-boundaries).

---

## Under the hood

```python
# calibrate / report / diff / check-artifact / gate are the omegaprompt CLI commands.
# run_p1 / P1Config come from the underlying omega-lock calibration engine:
from omega_lock import run_p1, P1Config   # run_p1 = the engine's core calibration run; the CLI wraps it
```

The calibration engine is [`omega-lock`](https://github.com/hibou04-ops/omega-lock), a parameter-calibration kernel hardened on quant-trading calibration before this prompt adapter existed. `omegaprompt` is the prompt-calibration CLI and PyPI distribution that wraps it with the held-out gate, three judges (`RuleJudge` / `LLMJudge` / `EnsembleJudge`), provider-neutral axes, the `CalibrationArtifact` schema (v2.0), the CI `diff`, and the new `gate` ship verdict.

The name *omega* is the final ship check — the last gate a prompt clears before it goes out. The name **omegaprompt** is earned at this point in the page, not assumed in the first screen.

> **Part of the omegaprompt toolkit** — [omegaprompt](https://github.com/hibou04-ops/omegaprompt) (calibration engine, this repo) · [omega-lock](https://github.com/hibou04-ops/omega-lock) (audit framework) · [antemortem-cli](https://github.com/hibou04-ops/antemortem-cli) (pre-implementation recon CLI) · [mini-omega-lock](https://github.com/hibou04-ops/mini-omega-lock) (empirical preflight) · [mini-antemortem-cli](https://github.com/hibou04-ops/mini-antemortem-cli) (analytical preflight) · [Antemortem](https://github.com/hibou04-ops/Antemortem) (methodology). Cross-toolkit cookbook (when-to-call-which-tool, 9 agent scenarios): [AGENT_TRIGGERS.md](AGENT_TRIGGERS.md).

---

## Go deeper

The rest of this README is the full reference: architecture, the `CalibrationArtifact` schema (v2.0), data contracts, all six search axes, the three judges, provider adapter details, validation, and limitations. New to it? Start with **[EASY_README.md](https://github.com/hibou04-ops/omegaprompt/blob/main/EASY_README.md)**. Browse worked tasks (code review, summarization, translation, debugging) in **[`examples/`](examples/)**.

License: Apache 2.0 · Copyright (c) 2026 hibou · PyPI: [omegaprompt](https://pypi.org/project/omegaprompt/) · CLI: `omegaprompt` (alias `omegacal`) · MCP: `omegaprompt-mcp`

---

## Table of contents

- [30-second demo (offline, no keys)](#30-second-demo--no-api-keys-no-network)
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
  - [10.1 Python (high-level API)](#101-python-high-level-api)
  - [10.2 CLI](#102-cli)
  - [10.3 MCP server (Claude Code, Cursor)](#103-mcp-server-claude-code-cursor)
- [11. Worked examples](#11-worked-examples)
- [12. Validation](#12-validation)
- [13. Comparative positioning](#13-comparative-positioning)
- [14. Limitations and scope boundaries](#14-limitations-and-scope-boundaries)
- [15. Roadmap](#15-roadmap)
- [16. Prior art and credits](#16-prior-art-and-credits)
- [Trust and toolkit docs](#trust-and-toolkit-docs)
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

2. **Execution profiles.** Strict mode (the `guarded` profile, default) refuses to silently relax validation — unship-grade judges raise, structured-schema fallback to prose raises, hidden capability loss raises. Fast mode (the `expedition` profile) permits controlled boundary crossing, but every relaxation is recorded as a `RelaxedSafeguard` entry on the artifact. The two profiles make the bargain between strictness and reach explicit and auditable.

3. **Capability tiers and explicit degradation events.** Each provider declares a `ProviderCapabilities` record (supports strict schema, json object, reasoning profiles, usage accounting, LLM judging, tools; tier CORE / CLOUD / LOCAL; experimental / placeholder flags). When an adapter degrades at runtime — for instance, retries without a rejected `reasoning_effort` parameter — it emits a `CapabilityEvent` capturing the capability, the requested value, the applied fallback, and a user-visible note. The event flows up through `EvalItemResult` → `EvalResult` → `CalibrationArtifact` so downstream diffs can detect capability regressions.

4. **Neutral-baseline vs calibrated comparison.** The `CalibrationArtifact` (schema v2.0) records the fitness of the neutral-parameter baseline and the fitness of the calibrated best side by side, with absolute and percent uplift, plus quality-per-cost and quality-per-latency ratios at both points. A reviewer sees not just *the best score* but *what the search earned over doing nothing*.

5. **Held-out ship gate with pre-declared thresholds.** The held-out test evaluation uses a Pearson-correlation threshold (`--min-kc4`) and a generalisation-gap threshold (`--max-gap`) that default from the execution profile and are recorded on the artifact. The thresholds cannot be lowered after the scores are seen; this is the Winchester defence, borrowed from quant finance. `status = FAIL_KC4_GATE` is a ship-blocker by construction.

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
│                  (Anthropic, OpenAI / OpenAI-compatible, Gemini,
│                  local/ollama/vllm/llama_cpp). Translates meta-axes to
│                  vendor parameters, reports capabilities and degradation.
├── judges/        Judge protocol + LLM / Rule / Ensemble implementations.
│                  Depends on domain and providers.
├── targets/       CalibrableTarget protocol + PromptTarget adapter. The
│                  composition point where omega-lock's search layer plugs in.
├── reporting/     Artifact → Markdown renderer.
├── commands/      Typer subcommands: calibrate, report, diff, check-artifact.
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
       │  factory, Anthropic/OpenAI/Gemini/local adapters)        │
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

The calibration *discipline* — sensitivity measurement, top-K unlock, grid search, held-out validation with pre-declared gates, hard-gate × soft-score fitness, artifact schema — is vendor-agnostic and lives in `core/` and `domain/`. The *adapter layer* — how a `reasoning_profile: deep` becomes a vendor-native API call, how a vendor's usage record is normalised to `input_tokens / output_tokens / cache_creation_input_tokens / cache_read_input_tokens` — lives in `providers/`. A reader evaluating the integrity of the calibration can review `core/` and `domain/` without caring which vendors are wired in. A reader onboarding a new provider can implement `LLMProvider` without reading the search layer.

---

## 4. Key abstractions

### 4.1 Meta-axes

Six axes constitute the public search space:

| Axis | Type | Semantics | Vendor-native translation example |
|---|---|---|---|
| `system_prompt_variant` | `int` | Index into `PromptVariants.system_prompts`. | Message-level system-prompt substitution. |
| `few_shot_count` | `int` | Count of examples from `PromptVariants.few_shot_examples`. | Message-list prefix length. |
| `reasoning_profile` | enum `OFF / LIGHT / STANDARD / DEEP` | How much reasoning effort the target should spend. | Anthropic `thinking={"type":"adaptive"}` + `effort` in `{low,medium,high}`; OpenAI `reasoning_effort`; Gemini currently records a `CapabilityEvent` for LIGHT/DEEP because no native mapping is used; local: system-prompt suffix. |
| `output_budget_bucket` | enum `SMALL / MEDIUM / LARGE` | Discretised `max_tokens`. | Resolved to `1024 / 4096 / 16000`. |
| `response_schema_mode` | enum `FREEFORM / JSON_OBJECT / STRICT_SCHEMA` | How strictly the response is shape-constrained. | Anthropic `messages.create` vs `messages.parse(output_format=...)`; OpenAI `chat.completions.create` vs `beta.chat.completions.parse(response_format=...)`; Gemini `generate_content` with `response_mime_type=application/json` and, for strict mode, `response_schema`. |
| `tool_policy_variant` | enum `NO_TOOLS / TOOL_OPTIONAL / TOOL_REQUIRED` | Tool-use policy. | No-op for plain chat targets; `tool_choice` derivative on tool-capable targets. |

The `MetaAxisSpace` record declares which values of each enum are in scope for a particular run; a single-member list locks the axis at a fixed value. The `ResolvedPromptParams` record carries the concrete choices after the searcher picks. Both records are Pydantic models with `extra="forbid"` — unknown keys raise at parse time.

### 4.2 Execution profiles

Two profiles capture the practitioner's position on the strict-versus-exploratory trade-off.

```python
class ExecutionProfile(str, Enum):
    GUARDED   = "guarded"     # default; blocks hidden fallbacks
    EXPEDITION = "expedition" # permits recorded boundary crossing
```

Strict mode (the `guarded` profile):
- Refuses to use a provider whose `supports_llm_judge` capability is false as a judge.
- Raises when a `STRICT_SCHEMA` request hits a provider that cannot honour it.
- Treats `experimental` or `placeholder` adapters as ineligible for ship-grade positions.
- Uses strict defaults for `max_gap` and `min_kc4` on the held-out gate.

Fast mode (the `expedition` profile):
- Permits the above, but every relaxation is recorded as a `RelaxedSafeguard` on the artifact, and `stayed_within_guarded_boundaries` is set to `False`.
- `additional_uplift_from_boundary_crossing` records how much of the calibrated fitness came from work that strict mode would have blocked, so the reviewer can see whether the boundary crossing actually paid off.

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
| `tier_2_cloud_grade` | First-class cloud providers; judge ship-grade is still declared separately. | Anthropic, OpenAI, Gemini. |
| `tier_3_local` | Local OpenAI-compatible backends. Target-eligible; by default not ship-grade judges. | Ollama, vLLM, llama.cpp, local OpenAI-compatible servers. |

Tiers are a policy input: strict mode refuses tier-3 providers in the judge position. Fast mode permits it, recording a `RelaxedSafeguard`.

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

Events propagate from the `ProviderResponse` up through the `EvalItemResult` into the `EvalResult.degraded_capabilities`, and finally onto `CalibrationArtifact.degraded_capabilities`. A reader of the artifact can grep for capability names to see which features were not honoured during the run. In strict mode, events with `affects_guarded_boundary=True` block the run; in fast mode they merely record.

### 4.5 Ship recommendations

The artifact's `ship_recommendation` field takes one of:

```python
class ShipRecommendation(str, Enum):
    SHIP = "ship"            # OK to deploy; all gates pass
    HOLD = "hold"             # do not deploy; at least one gate fails
    EXPERIMENT = "experiment" # opt-in expedition path; not a ship verdict
    BLOCK = "block"           # structural risk exceeds the current profile
```

Computation is deterministic from `status`, the held-out validation outcome, hard-gate pass rate, `stayed_within_guarded_boundaries`, and the presence of blocking `CapabilityEvent`s. Same artifact in, same recommendation out — a CI pipeline whitelists `SHIP` without interpreting prose. `omegaprompt diff` treats `BLOCK` and `HOLD` on the candidate as regressions even when the raw metrics improve; `EXPERIMENT` remains non-blocking by design.

---

## 5. The calibration pipeline

### 5.1 Inputs

Three files, all user-authored, all Pydantic-validated:

- **`dataset.jsonl`** — one `DatasetItem` per line: `id`, `input`, optional `reference`, optional `metadata`.
- **`rubric.json`** — `JudgeRubric` with per-dimension weight and integer scale, plus hard gates each labelled with an `evaluator` (`rule` / `judge` / `post`).
- **`variants.json`** — `PromptVariants` with `system_prompts` pool and optional `few_shot_examples`.

Optionally: `space.json` (custom `MetaAxisSpace`), `test.jsonl` (the held-out test slice the winner is re-tested on).

### 5.2 Sensitivity measurement

Around a neutral-parameter baseline, the searcher perturbs each meta-axis across its declared values and records the fitness delta. Axes are ranked by the Gini coefficient of their fitness-delta distribution — high Gini = concentrated, high-leverage; low Gini = diffuse, low-signal. Sensitivity is the *a priori* case for spending search budget on an axis.

### 5.3 Top-K unlock

The top `--unlock-k` axes by Gini delta enter the grid-search subspace. The rest stay locked at their neutral values. This cuts search cost from `Π(|axis|)` over all axes to `Π(|axis|)` over the top-K, typically a 5–20× reduction for `k=3`.

### 5.4 Grid search

Every combination in the unlocked subspace is evaluated. Each evaluation issues one provider call per dataset item (target) plus one judge call per item (if the judge is `LLMJudge` or `EnsembleJudge` with the LLM fallback triggered). The returned `EvalResult` records fitness, per-item scores, aggregate token usage, latency, and any capability events.

### 5.5 Held-out replay

The training-best parameters are replayed on the held-out test slice. The replay uses the *same* `PromptTarget` adapter with a different dataset; there is no leakage possible because the test slice was never seen by the searcher.

### 5.6 Held-out transfer gate (per-item correlation)

The Pearson correlation between train per-item scores and test per-item scores (on the shared dataset ids) is compared to `--min-kc4`. The generalisation gap `|train - test| / |train|` is compared to `--max-gap`. A failure on either sets `status = FAIL_KC4_GATE` and `ship_recommendation = HOLD`. Both thresholds are recorded on the artifact; they cannot be lowered after the fact.

**Transfer-gate semantics by validation_mode (v1.5+).** The transfer gate is a per-item correlation, so it is only meaningful when train and test slices share the *same* item ids — a "paired replay". On an ordinary disjoint train/test split the slices have no shared ids and the per-item correlation is structurally unmeasurable; the gate degenerates to the gap-only check. To make this explicit, `CalibrateTuning.validation_mode` accepts:

- `"auto"` (default, backward-compat): compute the per-item correlation only when slices share ≥3 ids; otherwise skip silently.
- `"paired"`: caller asserts shared ids by design. Raises `ValueError` if overlap < 3 — a paired run with no overlap is a setup bug, not a free pass.
- `"disjoint"`: caller asserts no shared ids by design. The per-item correlation is never computed; the gate is gap-only.

If you run a normal held-out split (disjoint ids), set `validation_mode="disjoint"` to make the artifact's lack of `kc4_correlation` self-documenting. Use `"paired"` when you score two prompts on the *same* items and want the per-item correlation as a stability signal.

### 5.7 Artifact emission

The `CalibrationArtifact` (see §8) is written as pretty-printed JSON to the `--output` path. It carries enough information to (a) render a Markdown report, (b) diff against a prior run, (c) gate CI on machine-readable fields without parsing prose.

### 5.8 Preflight and adaptation (optional sub-tool ecosystem)

The main pipeline does not assume that its default thresholds (`min_kc4 = 0.5`, `max_gap = 0.25`, `unlock_k = 3`) are universally correct. `omegaprompt.preflight` defines a stable plugin contract for two *optional* external sub-tools that measure the actual environment and emit a shared :class:`AdaptationPlan` the main pipeline consumes. The discipline's *defenses* — hard-gate fitness collapse, held-out ship gate, sensitivity-driven axis unlock — remain in place; only the numeric parameters are tuned to the environment.

**Standalone `omegaprompt` ships no preflight probe code.** Most users never need it. The preflight module exposes only:

- **Contracts** — Pydantic types (`PreflightReport`, `AnalyticalFinding`, `JudgeQualityMeasurement`, `EndpointMeasurement`, `PerformanceMeasurement`) that external sub-tools emit.
- **Adaptation logic** — `derive_adaptation_plan(report)` maps a report to an `AdaptationPlan`; `apply_adaptation_plan(plan, ...)` clips the plan against the caller's defaults so adaptation can only *strengthen* the discipline.

Two external sub-tools plug in:

| Sub-tool | Repository / PyPI | Role |
|---|---|---|
| **`mini-omega-lock`** | `pip install mini-omega-lock` (separate) | **Empirical preflight.** Probes the live judge + endpoint to measure consistency, schema reliability, context margin, latency, noise floor. Emits `JudgeQualityMeasurement`, `EndpointMeasurement`, `PerformanceMeasurement`. |
| **`mini-antemortem-cli`** | `pip install mini-antemortem-cli` (separate) | **Analytical preflight.** Reads the run configuration and classifies calibration trap patterns (self-agreement bias, small-sample transfer-gate power, rubric concentration, variant homogeneity, …) as `REAL` / `GHOST` / `NEW` / `UNRESOLVED`. Emits `AnalyticalFinding` records. |

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

Delegates to an `LLMProvider` via `ResponseSchemaMode.STRICT_SCHEMA`. Anthropic uses `messages.parse`, OpenAI uses `beta.chat.completions.parse`, and Gemini requests `response_schema` through the Google GenAI SDK before local Pydantic validation. No regex fallback, no prose-to-structure inference. Under strict mode, `LLMJudge` refuses to run on a provider whose `supports_llm_judge` capability is false; separate run-risk policy still treats non-ship-grade judges as a strict-mode-boundary issue.

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
| `gemini` | cloud | yes | yes | no | no | Google GenAI `generate_content`; supports target/freeform/json-object and strict schema via `response_schema` + local Pydantic validation. Not marked ship-grade as a strict-mode judge. |
| `ollama` / `local` / `vllm` / `llama_cpp` | local | best-effort | yes | no | no | Target-eligible; refuses LLM-judge position under strict mode. |

### 7.2 Anthropic

`messages.create` for freeform and JSON-object modes (with a system-prompt suffix instructing JSON output, since Anthropic does not expose a native `response_format={"type":"json_object"}`); `messages.parse(output_format=T)` for `STRICT_SCHEMA`. The system block is always wrapped with `cache_control={"type":"ephemeral"}` so repeated judge calls in a calibration run hit the prompt cache. Reasoning profiles map to `thinking={"type":"adaptive"}` plus `output_config.effort`.

### 7.3 OpenAI and OpenAI-compatible

`chat.completions.create` for freeform and `response_format={"type":"json_object"}` for JSON mode; `beta.chat.completions.parse(response_format=T)` for `STRICT_SCHEMA`. `reasoning_effort` is attempted for non-OFF reasoning profiles; when the endpoint rejects the parameter (some compatible endpoints do), the adapter retries without it and emits a `CapabilityEvent` naming the fallback. Accepts a `base_url`, which makes every OpenAI-compatible endpoint (Azure OpenAI, Groq, Together.ai, OpenRouter, local vLLM / Ollama) a drop-in target or judge.

### 7.4 Gemini

Uses the official Google GenAI SDK (`google-genai`). Freeform and JSON-object target calls are supported. `STRICT_SCHEMA` uses Gemini `response_schema` when enabled and still validates the response against the requested Pydantic model before returning. If native strict schema is unavailable, strict mode raises instead of degrading; fast mode may fall back to JSON-object output plus local Pydantic validation and records a `CapabilityEvent`.

Gemini is target-eligible for freeform and JSON-object runs, especially with `reasoning_profile` locked to `OFF` or `STANDARD`. `LIGHT` and `DEEP` reasoning profiles emit a `CapabilityEvent` because this adapter does not map them to a native Gemini control. Gemini can be used as a judge, but `ship_grade_judge=False`, so strict-mode artifacts should not be treated as ship-ready on that basis alone. Use Gemini judge paths in fast mode or independently validate judge reliability before changing capability policy.

### 7.5 Local endpoints

Local OpenAI-compatible backends are first-class target providers but are not considered ship-grade judges by default. Strict mode blocks their use in the judge position; fast mode records the relaxation. This is a policy position, not a library limitation — a local model that demonstrates ship-grade judge quality on your domain can have its capability override set explicitly.

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
  "neutral_fitness": "<float>",

  "calibrated_params": {
    "system_prompt_variant": 2,
    "few_shot_count": 1,
    "reasoning_profile": "deep"
  },
  "calibrated_fitness": "<float>",

  "best_params": { "...": "mirror of calibrated_params for backward-compat" },
  "best_fitness": "<float>",

  "uplift_absolute": "<float>",
  "uplift_percent": "<float>",
  "quality_per_cost_neutral": "<float>",
  "quality_per_cost_best": "<float>",
  "quality_per_latency_neutral": "<float>",
  "quality_per_latency_best": "<float>",

  "walk_forward": {
    "train_best_fitness": "<float>",
    "test_fitness": "<float>",
    "generalization_gap": "<float>",
    "validation_mode": "auto | paired | disjoint",
    "shared_item_count": "<integer>",
    "kc4_status": "COMPUTED | MISSING_PER_ITEM_SCORES | ...",
    "kc4_correlation": "<float-or-null>",
    "max_gap_threshold": "<float>",
    "min_kc4_threshold": "<float-or-null>",
    "passed": true
  },

  "hard_gate_pass_rate": "<float>",
  "sensitivity_ranking": [
    { "axis": "system_prompt_variant", "gini_delta": "<float>", "rank": 0 },
    { "axis": "reasoning_profile",     "gini_delta": "<float>", "rank": 1 },
    { "axis": "few_shot_count",        "gini_delta": "<float>", "rank": 2 }
  ],

  "boundary_warnings": [],
  "degraded_capabilities": [],
  "relaxed_safeguards": [],
  "stayed_within_guarded_boundaries": true,
  "additional_uplift_from_boundary_crossing": "<float>",
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

  "usage_summary": { "input_tokens": "<integer>", "output_tokens": "<integer>", "cache_read_input_tokens": "<integer>" },
  "latency_summary_ms": { "target_p50": "<float>", "judge_p50": "<float>" },
  "cost_basis": "normalized_token_units",
  "n_candidates_evaluated": "<integer>",
  "total_api_calls": "<integer>"
}
```

The key structural choice: `neutral_baseline` and `calibrated` are recorded side by side, with explicit `uplift_absolute` / `uplift_percent` fields. Exact values belong in generated artifacts or the claim ledger; the snippet above documents shape, not benchmark performance.

---

## 9. CLI surface

### 9.0 Exit-code contract

All CLI commands use the same contract:

- `0` — command completed and no requested gate/failure condition fired.
- `1` — CI gate failure: calibration status is non-OK, transfer-gate/hard-gate failure, `ship_recommendation` is `hold` or `block` in gate mode, artifact regression, or `check-artifact --strict` found integrity errors.
- `2` — environment/config/tooling/input problem: missing provider env var, unknown provider, missing dependency, invalid CLI argument, unreadable file, or invalid artifact passed to a non-checker command. Where applicable stderr uses explicit prefixes such as `TOOLING_MISSING`, `ENVIRONMENT_BLOCKED`, or `INVALID_ARTIFACT`.

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

Default behavior is CI-gate oriented. Exit codes: `0` when the artifact is CI-clean (`status == OK` and `ship_recommendation` is not `hold`/`block`), `1` when status is non-OK or `ship_recommendation` is `hold`/`block`, and `2` on CLI argument, environment, or tooling problems such as missing provider credentials, unknown providers, or missing dependencies. Use `--no-fail-on-gate` only when you want an advisory local run that still writes and prints the artifact.

### 9.2 `omegaprompt report <artifact.json>`

Renders a valid artifact as Markdown (for PR descriptions, CI step outputs, human review). It does not make a ship/no-ship decision: valid artifacts exit `0`; invalid artifact input exits `2`.

```bash
omegaprompt report artifact.json > report.md
```

### 9.3 `omegaprompt diff <old.json> <new.json>`

Compares two artifacts. Exits `1` when the new run regresses on any of: `calibrated_fitness`, `walk_forward.test_fitness`, `hard_gate_pass_rate`, `quality_per_cost_best`, `quality_per_latency_best`, `stayed_within_guarded_boundaries` (true-to-false is a regression), non-OK status, or `ship_recommendation` in `{hold, block}`. Invalid artifact input exits `2`. Intended for CI use; `--no-fail-on-regression` prints the regression but exits `0`.

```bash
omegaprompt diff previous.json artifact.json   # exit 1 on regression
```

### 9.4 `omegaprompt check-artifact <artifact.json>`

Runs the zero-network artifact integrity checker. Without `--strict`, it reports findings and exits `0` unless the file is inaccessible. With `--strict`, integrity errors exit `1`; inaccessible files / environment-blocked reads exit `2`. `--json` emits the machine-readable CI result.

The `omegacal` CLI binary remains as a compatibility alias during migration. `omegaprompt` is the primary CLI and PyPI distribution name; `omegacal` is not the PyPI distribution identity.

---

## 10. Quick start

```bash
pip install omegaprompt              # core (CLI + runtime entrypoints)
pip install "omegaprompt[mcp]"       # adds the MCP server for Claude Code / Cursor
```

omegaprompt exposes three calling surfaces over the same calibration kernel: a
**Python high-level API** (one call per operation, agent-callable), a **CLI**
(human-driven, scriptable from any shell), and an **MCP server** (Claude Code,
Cursor, and other MCP clients invoke the same operations as tools).

### 10.1 Python (high-level API)

Eight one-call entrypoints in `omegaprompt.runtime`, re-exported at package level.
Each accepts paths or inline objects and returns a Pydantic-modeled result.

```python
from omegaprompt import calibrate, evaluate, diff, report

artifact = calibrate(
    train="train.jsonl",
    test="test.jsonl",
    rubric="rubric.json",
    variants="variants.json",
    target="anthropic",                  # or {"name": "openai", "model": "gpt-4o"}
    judge="openai",                      # cross-vendor breaks self-agreement
    output="artifact.json",              # opt-in disk write
)
print(artifact.status, artifact.calibrated_fitness)

# Re-score the same config on a fresh dataset (regression check).
result = evaluate(
    dataset="canary.jsonl",
    rubric="rubric.json", variants="variants.json",
    params=artifact,                     # extracts artifact.calibrated_params automatically
    target="anthropic", judge="openai",
)

# Compare two artifacts (CI regression detection).
delta = diff("baseline.json", "candidate.json")
if delta.regressed:
    raise SystemExit("\n".join(delta.regression_reasons))

# Render for a PR description.
print(report("artifact.json"))
```

The four Tier 2 entrypoints — `measure_sensitivity`, `grade`, `preflight`,
`classify_traps` — cover diagnostic and per-response use cases. See
`omegaprompt/runtime.py` docstrings for the full surface.

Three input-coercion conveniences worth flagging:

- `target` and `judge` accept a string (`"anthropic"`), a `ProviderSpec` dict
  (`{"name": ..., "model": ..., "base_url": ...}`), or a pre-built `LLMProvider`
  instance. Use the string form for defaults; the dict form for non-default
  models or local endpoints; the instance form when you've already configured
  caching, retries, etc.
- Datasets, rubrics, and variants accept either a filesystem path or an
  in-memory Pydantic / dict instance. Agents typically pass dicts; humans
  typically pass paths.
- `params=` on `evaluate()` accepts a `CalibrationArtifact` directly. The
  common "evaluate the previous best on a new dataset" flow is one call.

Low-frequency knobs (search method, unlock-K, held-out gate thresholds, axis
space) are grouped under `tuning=CalibrateTuning(...)` rather than flat
parameters; the agent surface stays minimal while power users keep full
control.

### 10.2 CLI

```bash
# Anthropic target + Anthropic judge, guarded profile.
omegaprompt calibrate examples/sample_dataset.jsonl \
  --rubric examples/rubric_example.json \
  --variants examples/variants_example.json \
  --target-provider anthropic \
  --judge-provider anthropic \
  --profile guarded \
  --output artifact.json

# Cross-vendor (OpenAI target, Anthropic judge) to break self-agreement.
omegaprompt calibrate train.jsonl \
  --rubric rubric.json --variants variants.json --test test.jsonl \
  --target-provider openai   --target-model gpt-4o \
  --judge-provider anthropic --judge-model claude-opus-4-7 \
  --output artifact.json

# Local target (Ollama) + cloud judge.
omegaprompt calibrate train.jsonl \
  --rubric rubric.json --variants variants.json --test test.jsonl \
  --target-provider ollama \
    --target-base-url http://localhost:11434/v1 \
    --target-model llama3.1:8b \
  --judge-provider openai --judge-model gpt-4o \
  --profile guarded \
  --output artifact.json

# Gemini target + OpenAI judge.
export GEMINI_API_KEY=...
omegaprompt calibrate examples/sample_dataset.jsonl \
  --rubric examples/rubric_example.json \
  --variants examples/variants_example.json \
  --target-provider gemini \
  --target-model gemini-2.5-flash \
  --judge-provider openai \
  --output artifact.json

# Gemini judge path: explicit expedition profile.
omegaprompt calibrate examples/sample_dataset.jsonl \
  --rubric examples/rubric_example.json \
  --variants examples/variants_example.json \
  --target-provider anthropic \
  --judge-provider gemini \
  --judge-model gemini-2.5-flash \
  --profile expedition \
  --output artifact.json

# Render and diff.
omegaprompt report artifact.json
omegaprompt diff previous.json artifact.json
```

### 10.3 MCP server (Claude Code, Cursor)

The MCP server exposes all eight runtime entrypoints as agent-callable tools.
Inputs are JSON-friendly (paths, dicts, primitives); outputs are Pydantic models
serialized as dicts. Schema is auto-derived from type hints.

Run the server (stdio is the default; Claude Code spawns it as a subprocess):

```bash
pip install "omegaprompt[mcp]"
python -m omegaprompt.mcp           # stdio transport (recommended)
python -m omegaprompt.mcp --http    # streamable-http transport
```

Wire it into Claude Code's `mcpServers` configuration:

```json
{
  "mcpServers": {
    "omegaprompt": {
      "command": "python",
      "args": ["-m", "omegaprompt.mcp"]
    }
  }
}
```

Once the server is connected, the agent can call any of the eight tools by name
(`calibrate`, `evaluate`, `report`, `diff`, `measure_sensitivity`, `grade`,
`preflight`, `classify_traps`) with the documented argument shape. A typical
agent flow before shipping a prompt:

```
classify_traps(...)   →  catch self-agreement bias, small-sample power, etc.
preflight(...)         →  verify provider tiers, surface vendor-vendor warnings
calibrate(...)         →  run the full pipeline; produce a CalibrationArtifact
report(...)            →  render the artifact as Markdown for the user
diff(prev, new)        →  CI regression check on the next iteration
```

The MCP server is the substrate the rest of this README documents — every tool
is a thin wrapper around a runtime entrypoint, which in turn is a thin wrapper
around the calibration kernel.

For the canonical *when* — which trigger fires which tool flow across the four
MCP servers in the toolkit (omegaprompt, antemortem, mini-omega-lock,
mini-antemortem-cli, 18 tools total) — see [AGENT_TRIGGERS.md](AGENT_TRIGGERS.md).
That document maps real agent situations (non-trivial code change, prompt PR,
production regression suspected, agent self-verifying its output, etc.) to
the recommended cost-minimising tool sequence.

---

## 11. Worked examples

### 11.1 The failure mode the tool catches (illustrative)

A practitioner iterates prompts against a 10-item training set and compares two candidates.

```
Candidate A:  train_fitness = 0.923   (4.6 / 5 average)
Candidate B:  train_fitness = 0.876   (4.4 / 5 average)
```

Candidate A wins on training. The practitioner ships A.

Held-out validation on a test slice the searcher never saw:

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

The artifact records both candidates in the grid history, the failed held-out transfer gate, and `ship_recommendation: "hold"`. CI gating on `stayed_within_guarded_boundaries == true` and `ship_recommendation == "ship"` blocks the merge without the practitioner needing to parse prose.

> Numbers in this subsection are illustrative of the failure mode. A reproducible, machine-generated example is in §11.2.

### 11.2 Reproducible reference run

The repository ships a deterministic reference run (`examples/reference/reproduce_reference_artifact.py`) that drives the **real** `omega_lock.run_p1` against an in-memory target + judge whose fitness is a closed-form function of the meta-axis parameters. No LLM API calls, no network, no randomness. The same run produces byte-identical output on every machine.

To reproduce and check the offline golden harness:

```bash
python examples/reference/reproduce_reference_artifact.py
python tools/reproduce_golden_reference.py --check
omegaprompt check-artifact examples/reference/reference_artifact.json --strict
```

The harness writes `examples/reference/golden_manifest.json`, which records each case id, reproducible command, expected status, expected ship recommendation, expected validation mode, integrity classification, normalized artifact hash, and whether exact metrics may be displayed. It covers:

- `clean_ok_ship` — the real `omega_lock.run_p1` path with deterministic target and judge stubs.
- `fail_kc4_gate` — paired held-out validation with the per-item transfer correlation below the pre-declared threshold.
- `fail_hard_gates` — hard-gate failure with a BLOCK recommendation.
- `provider_degradation` — explicit capability degradation and relaxed safeguard visibility.
- `diff_regression_candidate` — an individually valid artifact that regresses against the clean baseline under `omegaprompt diff`.

The real calibration engine is not mocked for the clean case. `omega_lock.run_p1` runs with production search logic; only the *target* and *judge* layers are deterministic stubs so the output is reproducible without API access. Exact values belong in the generated artifact, golden manifest, or claim ledger rather than hand-maintained README prose.

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

The default test suite runs with `pytest -q` and uses mocked provider clients rather than live provider/API calls. Exact test counts are intentionally not repeated in README prose; the static badge row is preserved separately and protected by the consistency checker. Adapter tests use `SimpleNamespace` or `MagicMock` in place of SDK clients and assert the outgoing request payload shape (model, messages, cache headers, `response_format`, reasoning directives, few-shot ordering). The sub-tool repositories `mini-omega-lock` and `mini-antemortem-cli` carry their own test suites covering probe execution and analytical trap classification respectively.

| Module | Coverage summary |
|---|---|
| `domain/` | `PromptVariants` / `MetaAxisSpace` / `CalibrationArtifact` / `Dataset` / enums / profiles — required fields, range validation, JSON roundtrip, ordinal clamping, compat-key mapping, `model_post_init` synchronisation between `best_params` and `calibrated_params`. |
| `core/fitness` | `CompositeFitness` over empty / all-pass / partial-fail / all-fail batches; per-item breakdown preserved for reporting. |
| `core/walkforward` | Pearson correlation over shared ids; zero-variance skip (kc4=None); gap arithmetic; gate pass/fail logic for both thresholds. |
| `core/sensitivity` | Gini-coefficient ranking; top-K unlock; edge cases (zero-delta axes, single-point probes). |
| `core/artifact` | Round-trip through JSON on disk; `model_post_init` invariants on load. |
| `core/profiles` | `policy_for(GUARDED/EXPEDITION)` returns distinct defaults; `relaxed_safeguards_for(...)` reports crossings. |
| `core/risk` | `assess_run_risk(...)` across OK / transfer-gate fail / hard-gate fail / capability-event scenarios. |
| `providers/` | Factory rejects unknown names; respects `base_url`; lists `anthropic` / `openai` / `gemini` / `ollama`. Anthropic adapter: freeform uses `messages.create` with thinking config when reasoning enabled, omits it when OFF; strict schema uses `messages.parse`; refusal raises; JSON-object mode adds system-prompt suffix. OpenAI adapter: same paths on `chat.completions.create` / `beta.chat.completions.parse`; `reasoning_effort` rejected-retry records `CapabilityEvent`; `prompt_tokens_details.cached_tokens` normalised to `cache_read_input_tokens`; content-filter finish reason raises; missing `parsed` raises. Gemini adapter: freeform/json-object/strict-schema request shapes, strict-mode no-degrade rule, fast-mode JSON fallback with `CapabilityEvent`, malformed/schema-invalid JSON failures, and usage mapping. `ollama` alias reports tier `tier_3_local`, `supports_llm_judge=False`, `experimental=True`. |
| `judges/` | `RuleJudge` (no_refusal / non_empty / json_object / regex / duplicate-check rejection / missing-check raise); `LLMJudge` (strict-schema dispatch, payload composition, non-`JudgeResult` response raise, strict-mode ship-grade judge refusal); `EnsembleJudge` (rule-first short-circuit, LLM escalation on rule-pass, merged gate_results, non-`RuleJudge` rejection). |
| `targets/` | `PromptTarget` end-to-end with mocked provider + judge; meta-axis resolution and clamping for out-of-range inputs; usage accumulation across evaluations; `evaluation_history` retention; latency measurement; degraded-capability propagation. |
| `commands/` | CLI help lists `calibrate` / `report` / `diff` / `check-artifact`; `--version` shows `omegaprompt`; `report` renders schema-v2.0 artifacts; `diff` detects regressions on fitness, cost ratios, latency ratios, boundary-crossing flips; `check-artifact` performs zero-network integrity checks. |
| `preflight/` | **Plugin interface only** — no probe or classifier code inside `omegaprompt`. `contracts`: severity ordering, status enum, `PreflightReport.worst_severity` / `any_real_or_new`, Pydantic `extra="forbid"` enforcement; bounds on `JudgeQualityMeasurement.consistency` (0..1), `EndpointMeasurement.schema_reliability` (0..1). `adaptation`: noise-adaptive `min_kc4` across four thresholds; consistency-driven `rescore_count`; schema-fallback trigger; wall-time-driven `unlock_k` reduction; small-sample gap widening; variant-skip axis marking; `apply_adaptation_plan` invariants (never weakens `min_kc4`, never widens `max_gap`, never raises `unlock_k`). Sub-tool probe + classifier implementations (with their own test suites) live in the `mini-omega-lock` and `mini-antemortem-cli` repositories. |
| `test_calibrate_integration.py` | **Drives the real `omega_lock.run_p1`** with a deterministic in-memory `CalibrableTarget` (no mocks on the search engine). Asserts the artifact's `calibrated_params`, `neutral_baseline_params`, `walk_forward`, and `sensitivity_ranking` match `P1Result`'s actual shape — the regression that this test catches is drift between the adapter layer and the search engine that per-module unit tests cannot reach. |

Run the default no-network suite with `uv run pytest -q -m "not live"` (or `python -m pytest -q -m "not live"`). The wall-clock time is dominated by Pydantic model compilation on first import.

---

## 13. Comparative positioning

| Approach | What it does well | What `omegaprompt` adds |
|---|---|---|
| **promptfoo** | Runs prompts against test cases with assertion-based grading. | Pre-declared held-out gate, sensitivity-ranked axis unlock, `hard_gate × soft_score` fitness, machine-readable diffable artifact. Composable — promptfoo-style assertions plug in as `RuleJudge` checks. |
| **DSPy** | Prompt synthesis via program abstraction + bootstrapped few-shot. | Orthogonal concern. DSPy *produces* candidate prompts; `omegaprompt` *decides which one ships* after held-out validation. DSPy outputs plug in as `system_prompts` entries in `PromptVariants`. |
| **Optuna / Ray Tune on prompts** | Generic hyperparameter optimisation. | Held-out ship gate and pre-declared kill criteria out of the box; schema-enforced LLM-as-judge via each vendor's native parse path; provider-neutral meta-axes; explicit `CalibrationArtifact` schema CI can diff. |
| **Provider-native evaluation dashboards** | Rubric-based grading inside one vendor's console. | Cross-vendor judging (break self-agreement bias); local artifact that does not require vendor login; deterministic `diff` for regression detection; fast mode for controlled boundary crossing. |
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

The LLM judge's scoring distribution can drift across model releases. A planned multi-judge validation pattern (`judge_v1` vs `judge_v2` on the top-K) treats disagreement as a trust signal rather than a silent failure.

### Cost is non-trivial

A typical run (10-item dataset, 125-candidate grid, held-out validation) on frontier-tier cloud providers costs in the tens of dollars. Mitigations: cheaper judge during iteration, prompt-cache-aware scheduling within a 5-minute window, local target via Ollama when quality permits.

### Not all providers are ship-grade judges

Strict mode blocks local providers in the judge position by policy. Gemini is implemented and can validate `JudgeResult`, but is not marked ship-grade as a strict-mode judge in this adapter. Use it as a target freely; use it as a judge in fast mode or after independent domain validation and a deliberate capability policy change.

---

## 15. Roadmap

**Shipped (v1.0)**
- Provider-neutral meta-axes (`reasoning_profile`, `output_budget_bucket`, `response_schema_mode`, `tool_policy_variant`).
- Unified `LLMProvider.call(ProviderRequest) -> ProviderResponse` + `capabilities()` contract.
- `ExecutionProfile` (strict mode / fast mode) + structural risk reporting.
- `CalibrationArtifact` schema v2.0 (neutral baseline vs calibrated, capability events, boundary warnings, ship recommendation).
- `RuleJudge` / `LLMJudge` / `EnsembleJudge`.
- Native `gemini` adapter + `ollama` / `local` / `vllm` / `llama_cpp` local adapter family.
- CLI: `calibrate` / `report` / `diff` / `check-artifact`. Backward-compat `omegacal` alias.
- Integration test against real `omega_lock.run_p1`.

**Planned: judge trust + tooling depth**
- Multi-judge validation pattern: `judge_v1` + `judge_v2` over top-K; disagreement as a first-class trust signal.
- `--dry-run` with cost estimate before launching.
- Additional rule-gate predicates (language detection, length bounds, schema validation against a supplied JSON Schema).

**Planned: ecosystem**
- Benchmark harness: multi (task × rubric × seed) scorecards.
- GitHub Action for CI regression gating via `omegaprompt diff`.
- HTML report rendering (`omegaprompt report --format html`).
- Native HuggingFace Inference adapter.

**Explicitly out of scope**
- Hosted dashboard, database-backed history, multi-tenant service. `omegaprompt` is a local developer tool. Keep it local.

Full changelog: [CHANGELOG.md](CHANGELOG.md).

---

## 16. Prior art and credits

- **Train / test split with a pre-declared gate.** The foundational ML defence against overfitting, documented in every undergraduate curriculum. The specific implementation here (Pearson rank correlation threshold, pre-declared and unmodifiable) is the held-out transfer gate — the kill criterion the `omega-lock` engine names `KC-4` internally.
- **LLM-as-judge.** Pattern formalised in *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (Zheng et al., 2023). `omegaprompt` implements the pattern with schema enforcement at the SDK boundary (Pydantic via each vendor's native parse path) so malformed judge responses raise before polluting the fitness.
- **Winchester defence.** A quant-finance discipline: *kill criteria declared before the run cannot be relaxed after.* Used here to argue that `--max-gap` and `--min-kc4` must be enforced in configuration, not retroactively tuned on inspection of scores.
- **Sensitivity-driven coordinate descent.** Stress measurement and top-K unlock are the parameter-calibration primitives introduced by `omega-lock` (v0.3.0), originally for trading-strategy calibration, ported here to prompt configuration.
- **Antemortem discipline.** The pre-implementation reconnaissance methodology under which this project was designed and built. Every non-trivial change runs through [`antemortem-cli`](https://github.com/hibou04-ops/antemortem-cli) before the first keystroke. The case studies in the [methodology repository](https://github.com/hibou04-ops/Antemortem) record the recon for this codebase.

Naming: *omega-lock* (parameter calibration) → *omegaprompt* (prompt calibration). The family branding is intentional. `omega-lock` was extracted from a trading-strategy calibration that ended in `KC-4 FAIL` — the overfitting defence firing exactly as designed. `omegaprompt` is the same defence applied one layer up, and the sub-tools `mini-omega-lock` / `mini-antemortem-cli` extend the pattern to preflight measurement.

---

## Troubleshooting

### `omegaprompt calibrate` returns "Incorrect API key" / 401

The provider SDK got a key, but the key was invalid. Two common cases:

- **Stale or revoked key.** Check the issuing dashboard (Anthropic / OpenAI / Google AI Studio). Rotate and re-export.
- **Wrong env var.** Each provider reads its own variable. The CLI does *not* fall back across vendors:

  | Provider | Accepted env vars |
  |---|---|
  | `anthropic` | `ANTHROPIC_API_KEY` |
  | `openai` | `OPENAI_API_KEY` |
  | `gemini` | `GEMINI_API_KEY` **or** `GOOGLE_API_KEY` (first non-empty wins) |
  | `local` / `ollama` / `vllm` / `llama_cpp` | none required (uses `--base-url`) |

  Setting `OPENAI_API_KEY` does not authenticate `--target-provider gemini`. Setting `GEMINI_API_KEY` does not authenticate `--target-provider openai`.

### "ProviderError: Gemini API key is required for provider='gemini'"

Neither `GEMINI_API_KEY` nor `GOOGLE_API_KEY` is set. Get a free tier key at <https://aistudio.google.com/apikey>, then:

```bash
export GEMINI_API_KEY=AIza...
```

### How do I sanity-check before spending API budget?

Run the deterministic smoke test — no keys, no network:

```bash
python examples/reference/reproduce_reference_artifact.py
omegaprompt report examples/reference/reference_artifact.json
```

This exercises the calibration kernel + judge + artifact serialization end-to-end without ever touching a provider. If this fails, the install is broken; if it passes, the providers are the next thing to test.

For a single live call per provider (smallest possible spend), construct a `ProviderRequest` directly:

```python
from omegaprompt.providers.factory import make_provider
from omegaprompt.providers.base import ProviderRequest

req = ProviderRequest(system_prompt="Be brief.", user_message="Say OK.")
for name in ("anthropic", "openai", "gemini"):
    try:
        resp = make_provider(name).call(req)
        print(name, "OK", resp.usage)
    except Exception as e:
        print(name, "FAIL", e)
```

A 401 here is a key issue; an `ImportError` is a missing optional vendor SDK (`pip install "omegaprompt[anthropic]"` etc.); a successful call confirms the provider path is healthy and the eval can proceed.

### Gemini call works but `LLMJudge` refuses under strict mode

By design. Gemini is `ship_grade_judge=False`. Under strict mode the judge tier check fails fast rather than producing an artifact whose ship recommendation rests on an unvalidated judge. Two ways forward:

- Use Gemini as the **target** and Anthropic / OpenAI as the **judge** (cross-vendor still satisfies strict mode).
- Run under `--profile expedition`, which records a `RelaxedSafeguard` rather than failing — the artifact will reflect the relaxed boundary and downstream `diff` will surface it.

Validate independently before flipping `ship_grade_judge=True` in any forked adapter.

---

## Trust and toolkit docs

The trust-heavy details live in focused docs so README prose stays source-backed:

- [Trust model](docs/trust-model.md) — what a `CalibrationArtifact` proves, what it does not prove, train/test discipline, held-out transfer-gate limits, offline vs live evidence, the no-live-provider default CI rule, MCP optional boundary, and diff regression use.
- [Toolkit positioning](docs/toolkit-positioning.md) — `omegaprompt` vs `omega-lock`, `antemortem-cli`, optional `mini-*` preflight plugins, the `omegacal` compatibility alias, and the no-dashboard/no-web-app scope.
- [Provider capabilities](docs/provider-capabilities.md) — adapter capability claims tied to code and contract tests.
- [Profiles and risk boundaries](docs/profiles-and-risk-boundaries.md) — strict mode vs fast mode behavior and validation-mode interpretation.

Exact public claims remain governed by the claim ledger and deterministic artifacts.

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
    EXPERIMENT = "experiment"
    BLOCK = "block"

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

| Meta-axis value | Anthropic | OpenAI / compatible | Gemini | Local (Ollama / vLLM / llama.cpp) |
|---|---|---|---|---|
| `reasoning_profile = OFF` | no `thinking` block | no `reasoning_effort` | model default | system prompt unchanged |
| `reasoning_profile = LIGHT` | `thinking={type:adaptive}` + `effort: low` | `reasoning_effort: low` (if supported) | model default + `CapabilityEvent` | system-prompt suffix: "think briefly" |
| `reasoning_profile = STANDARD` | `thinking={type:adaptive}` + `effort: medium` | `reasoning_effort: medium` | model default | system-prompt suffix: "think step by step" |
| `reasoning_profile = DEEP` | `thinking={type:adaptive}` + `effort: high` | `reasoning_effort: high` | model default + `CapabilityEvent` | system-prompt suffix: "think carefully step by step" |
| `output_budget_bucket = SMALL` | `max_tokens=1024` | `max_tokens=1024` | `max_output_tokens=1024` | `max_tokens=1024` |
| `output_budget_bucket = MEDIUM` | `max_tokens=4096` | `max_tokens=4096` | `max_output_tokens=4096` | `max_tokens=4096` |
| `output_budget_bucket = LARGE` | `max_tokens=16000` | `max_tokens=16000` | `max_output_tokens=16000` | `max_tokens=16000` |
| `response_schema_mode = FREEFORM` | `messages.create` | `chat.completions.create` | `generate_content` | `chat.completions.create` |
| `response_schema_mode = JSON_OBJECT` | `messages.create` + system-prompt JSON suffix | `response_format={type:json_object}` | `response_mime_type=application/json` + JSON suffix | best-effort system-prompt instruction |
| `response_schema_mode = STRICT_SCHEMA` | `messages.parse(output_format=T)` | `beta.chat.completions.parse(response_format=T)` | `response_schema=T` + local Pydantic validation; if unavailable, fast-mode-only JSON fallback | not supported; strict mode raises |
| `tool_policy_variant = NO_TOOLS` | no `tools` argument | no `tools` argument | no `tools` argument | no `tools` argument |
| `tool_policy_variant = TOOL_OPTIONAL` | `tools=[...]`, no `tool_choice` | `tools=[...], tool_choice="auto"` | not mapped | adapter-specific |
| `tool_policy_variant = TOOL_REQUIRED` | `tools=[...], tool_choice={type:"any"}` | `tools=[...], tool_choice="required"` | not mapped | adapter-specific |

Any cell that reads "not supported" or "best-effort" emits a `CapabilityEvent` at runtime and, under strict mode, may block the run depending on the execution profile policy.

---

## Appendix C: invariants

The following properties hold by construction and are enforced either in the Pydantic schema layer or in a dedicated test. They are the "theorems" a reviewer can rely on without reading the implementation.

1. **No client-side schema regex.** `STRICT_SCHEMA` mode dispatches to the provider's strongest structured path (`messages.parse` on Anthropic, `beta.chat.completions.parse` on OpenAI, `response_schema` on Gemini when available). Any fallback to JSON-object plus local Pydantic validation is explicit and fast-mode-only. A malformed structured response raises before the calibration loop sees it.
2. **Hard-gate fitness collapse.** For any `(item, params)` pair, if any `hard_gate` returns `False`, the item's contribution to `CompositeFitness` is `0.0`. No soft penalty, no partial credit.
3. **Held-out threshold immutability.** `--max-gap` and `--min-kc4` are CLI arguments resolved once at run start and recorded on the artifact. There is no API surface for modifying them mid-run.
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
omegaprompt v2.0.2 — provider-neutral prompt calibration engine.
https://github.com/hibou04-ops/omegaprompt, 2026.
```

BibTeX:

```bibtex
@software{omegaprompt_2026,
  author  = {hibou04-ops},
  title   = {{omegaprompt}: Provider-neutral prompt calibration engine
             with sensitivity-ranked meta-axes, walk-forward ship gates,
             and structural capability reporting},
  version = {2.0.2},
  year    = {2026},
  url     = {https://github.com/hibou04-ops/omegaprompt}
}

@software{omegalock_2026,
  author  = {hibou04-ops},
  title   = {{omega-lock}: Sensitivity-driven coordinate-descent
             calibration framework with walk-forward validation and
             pre-declared kill criteria},
  version = {0.3.0},
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

**License history.** PyPI distributions of versions 1.1.0 and earlier were shipped with an MIT `LICENSE` file. The repository was relicensed to Apache 2.0 between the 1.1.0 and 1.2.0 PyPI uploads; 1.2.0 (2026-04-27) and all later versions ship under Apache 2.0. Anyone who installed 1.1.0 or earlier holds an MIT license to that copy — license changes do not apply retroactively.

## Colophon

Designed, implemented, and shipped solo. Adapter layer over `omega-lock`; zero calibration-engine reimplementation. Every non-trivial change is pre-authored through `antemortem-cli`'s recon discipline. Tests run offline; no live API calls in CI.
