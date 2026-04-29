# Agent Trigger Guide

> **Purpose**: When should an agent (Claude Code, Cursor, custom runtime) call which MCP tool from this toolkit? This document maps real agent situations to tool flows. Read it as the cookbook the README's MCP section references.

> **Scope**: Four MCP servers across the toolkit, 18 tools total.
> - `omegaprompt.mcp` — 8 tools (calibrate, evaluate, report, diff, measure_sensitivity, grade, preflight, classify_traps)
> - `antemortem.mcp` — 3 tools (scaffold, run, lint)
> - `mini_omega_lock.mcp` — 5 tools (empirical_preflight + 4 individual probes)
> - `mini_antemortem_cli.mcp` — 2 tools (analytical_preflight, list_traps)

The agent does not need to know the calibration discipline; it needs to know **when to invoke which tool**. Each scenario below names a trigger, the tool flow, and what to do with the output.

---

## Quick reference

| Trigger | First tool | Then |
|---|---|---|
| Non-trivial code change requested | `antemortem.scaffold` | run, lint |
| Prompt config edited | `mini_antemortem_cli.analytical_preflight` | preflight, classify_traps, calibrate |
| Pre-ship validation | `omegaprompt.calibrate` | report |
| PR with prompt changes | `omegaprompt.evaluate` | diff |
| Production regression suspected | `omegaprompt.evaluate` (canary) | diff (vs baseline) |
| Agent self-verifying output | `omegaprompt.grade` (strategy="rule") | — |
| Wondering if calibrate is worth cost | `omegaprompt.measure_sensitivity` | — |
| New environment / endpoint | `omegaprompt.preflight` | empirical_preflight |
| Long calibration about to start | `mini_omega_lock.project_performance` | — |

---

## Scenario 1 — Pre-implementation reconnaissance

**Trigger.** Agent is asked to make a non-trivial change to a codebase: refactor a module, add a feature that touches several files, modify auth or migrations, anything where the agent might confidently produce a plausible-but-wrong patch.

**Why this matters.** LLM agents are most likely to ship bugs when they confidently execute on an under-specified problem. The Antemortem discipline pulls hidden risks into a structured document **before** the patch is written.

**Tool flow.**

```
antemortem.scaffold(name, enhanced)        # creates antemortem/<name>.md
  ↓ (agent fills in spec, traps, files_to_read)
antemortem.run(document, provider, ...)    # LLM classifies risks against actual repo files
  ↓
antemortem.lint(document)                  # verifies file:line citations are not hallucinated
  ↓
read decision                              # SAFE_TO_PROCEED / PROCEED_WITH_GUARDS / NEEDS_MORE_EVIDENCE / DO_NOT_PROCEED
```

**What to do with the output.**

- `SAFE_TO_PROCEED` — proceed with the change as planned.
- `PROCEED_WITH_GUARDS` — proceed, but apply the listed remediations first or in tandem.
- `NEEDS_MORE_EVIDENCE` — gather more context (read more files, ask the user) before continuing.
- `DO_NOT_PROCEED` — stop. The classification flagged an unmitigated high-severity REAL risk.

**Cost.** One LLM call (~3-15K output tokens). The `lint` step is deterministic (zero-cost). Optional `--critic` doubles the cost but downgrades weakly-supported REAL findings.

---

## Scenario 2 — Pre-calibration sanity check

**Trigger.** Agent is about to call `omegaprompt.calibrate` (an expensive operation, dozens to hundreds of LLM calls). Cheap-to-fix configuration issues should be caught first.

**Why this matters.** A misconfigured calibration is costlier than a refused one. Self-agreement bias (target and judge from the same vendor), small-sample power loss, variant homogeneity — these all *will* produce numbers, but the numbers won't tell you what you want.

**Tool flow.**

```
mini_antemortem_cli.analytical_preflight(target_provider, judge_provider, train_dataset, rubric, variants)
  ↓ (returns list of AnalyticalFinding; REAL/NEW labels = real problems)
omegaprompt.preflight(target, judge, profile)
  ↓ (capability check, surfaces self-agreement-bias warnings)
[optional] mini_omega_lock.empirical_preflight(rubric, probe_item, probe_response, provider, ...)
  ↓ (judge consistency, endpoint reliability, latency probe)
proceed to calibrate, or fix the surfaced issues first
```

**What to do with the output.**

- `analytical_preflight` REAL/NEW findings — fix before running calibrate.
- `preflight` `status="abort"` — do not proceed; placeholder provider or other blocker.
- `empirical_preflight` `judge_quality.consistency < 0.7` — pick a stronger judge or change provider.
- `empirical_preflight` `endpoint.silent_degradation_detected` — check provider settings.

**Cost.** `analytical_preflight` is deterministic (zero LLM cost). `preflight` is capability-only (zero LLM cost). `empirical_preflight` costs `consistency_repeats` LLM calls (default 3).

---

## Scenario 3 — Pre-ship validation (the full gate)

**Trigger.** Prompt is about to be deployed to production. Agent (or CI bot) runs the full validation pipeline.

**Tool flow.**

```
mini_antemortem_cli.analytical_preflight(...)     # cheap config sanity
omegaprompt.preflight(...)                         # cheap env sanity
omegaprompt.classify_traps(...)                    # same as analytical_preflight when mini-* installed
[optional] omegaprompt.measure_sensitivity(...)    # cheap probe — should we calibrate at all?
omegaprompt.calibrate(train, test, rubric, variants, target, judge)
  ↓
omegaprompt.report(artifact)                       # markdown for PR description / chat
  ↓
inspect artifact.status:
  - OK                  → ship
  - FAIL_KC4_GATE       → walk-forward gate failed; do not ship
  - FAIL_HARD_GATES     → structural risk exceeded profile boundary; do not ship
```

**What to do with the output.**

- `OK` — ship. Attach `report` markdown to the PR / changelog.
- `FAIL_KC4_GATE` — the calibrated prompt overfit to the training set. Either expand the training set, regularize the search, or accept the hold-out failure and ship the neutral baseline.
- `FAIL_HARD_GATES` — provider capability or self-agreement bias exceeded the guarded-profile limit. Switch judge vendor or use `expedition` profile knowingly.

**Cost.** `calibrate` is the expensive call (dozens to hundreds of LLM calls). Everything before it is cheap or free.

---

## Scenario 4 — PR with prompt changes (CI regression check)

**Trigger.** A pull request modifies a prompt configuration (variants.json, rubric.json, or the system prompt itself). CI bot runs an evaluation against the existing baseline artifact.

**Tool flow.**

```
omegaprompt.evaluate(dataset, rubric, variants, params, target, judge)
  ↓ (scores the new variants against the held-out test dataset)
omegaprompt.diff(baseline_artifact, candidate_artifact)
  ↓
inspect diff:
  - regressed=false    → merge OK
  - regressed=true     → block merge; surface regression_reasons in PR
```

**What to do with the output.**

- `regressed=false` and `fitness_delta >= 0` — approve the PR's prompt change.
- `regressed=true` — block; the PR template should display `diff(format="markdown")` so the human reviewer sees which metric dropped.

**Cost.** `evaluate` costs roughly `dataset_size × 2` LLM calls. Cheaper than a full calibrate.

---

## Scenario 5 — Production regression suspected

**Trigger.** Production quality metric drops, user reports inconsistent outputs, or a scheduled canary check fires.

**Tool flow.**

```
omegaprompt.evaluate(canary_dataset, rubric, variants, params=current_artifact, target, judge)
  ↓
omegaprompt.diff(baseline_artifact, current_canary_eval)
  ↓
if regressed:
  inspect canary_eval.degraded_capabilities
    - has CapabilityEvent?  → provider silently dropped a feature; fix env / pin provider version
    - clean?                → judge or target model upgraded behind the API; recalibrate
```

**What to do with the output.**

- Capability event present (`reasoning_profile dropped`, `strict_schema rejected`, etc.) — the provider changed something. Either pin the SDK version or adapt.
- No capability events but fitness still dropped — the model itself drifted. Re-run `calibrate` to re-tune for the new behavior.

**Cost.** Same as Scenario 4 — one `evaluate` per canary run.

---

## Scenario 6 — Agent self-verifying its own output

**Trigger.** Agent has just produced a response (a code patch, a summary, an answer) and wants a quick quality check before returning it to the user.

**Tool flow.**

```
omegaprompt.grade(rubric, item, response, provider, strategy="rule")
  ↓
inspect JudgeResult:
  - hard_gate_pass_rate = 1.0   → return response to user
  - hard_gate_pass_rate < 1.0   → response failed a deterministic gate; regenerate
```

**What to do with the output.**

- All hard gates pass — the response is safe to return.
- Some hard gate fails — the response triggered a deterministic check (refusal pattern, empty content, malformed JSON). Regenerate or apply the gate's remediation before returning.

**Use `strategy="rule"`** for the cheap deterministic check (zero API cost). Use `strategy="ensemble"` if rule passes but you want the LLM judge's qualitative score as well.

**Cost.** `strategy="rule"` is zero-cost. `strategy="ensemble"` is one extra LLM call **only if** rule gates pass.

---

## Scenario 7 — "Should I bother calibrating?"

**Trigger.** Agent or developer is unsure whether `calibrate` will produce meaningful uplift on a given task. Calibration is expensive; running it on a task with no signal is wasted budget.

**Tool flow.**

```
omegaprompt.measure_sensitivity(dataset, rubric, variants, target, judge)
  ↓
inspect SensitivityResult.rows:
  - top axis normalized_stress > 0.3  → calibrate is worth it; 1-3 axes carry signal
  - all axes normalized_stress < 0.1  → no axis varies fitness meaningfully; skip calibrate
```

**What to do with the output.**

- High-stress axes — proceed to `calibrate`. Optionally pass `tuning=CalibrateTuning(unlock_k=k)` matching the count of high-signal axes.
- All-low stress — calibrate will not produce uplift. Either improve the rubric (so it discriminates), expand the variants (so there's something to vary), or accept the neutral baseline as good enough.

**Cost.** Roughly `axes × probe_size` LLM calls — typically 6-30 calls against a small slice. Far cheaper than a full calibrate.

---

## Scenario 8 — Performance / cost projection

**Trigger.** A long-running calibration is about to start. The agent (or developer) wants to know how long and how expensive before committing.

**Tool flow.**

```
mini_omega_lock.compute_context_margin(...)         # deterministic; will inputs fit context?
  ↓
[run a few cheap probe calls outside the toolkit, collect latencies_ms]
  ↓
mini_omega_lock.project_performance(probe_latencies_ms, dataset_size, candidates_expected)
  ↓
inspect:
  - projected_wall_time_seconds > 1 hour?  → reduce dataset_size or unlock_k
  - context margin negative                 → reduce variants / shorten system prompt
```

**Cost.** Both tools are deterministic — zero LLM cost.

---

## Scenario 9 — Onboarding a new vendor or endpoint

**Trigger.** Agent is configured with a new provider (e.g., switching from Anthropic to a local Ollama endpoint, or adding Groq for faster inference). Want to verify the new endpoint is ship-grade before pointing real workloads at it.

**Tool flow.**

```
omegaprompt.preflight(target=new_provider, judge=cross_vendor_judge, profile="guarded")
  ↓ (capability tier, placeholder check, self-agreement warnings)
mini_omega_lock.empirical_preflight(rubric, probe_item, probe_response, provider=new_provider, ...)
  ↓ (judge consistency, endpoint reliability, latency)
inspect:
  - preflight.status = abort         → do not use this endpoint as-is
  - empirical.endpoint.silent_degradation_detected = true → fix env first
  - empirical.judge_quality.consistency < 0.7 → not a viable judge
```

**Cost.** `preflight` is zero-cost; `empirical_preflight` is `consistency_repeats` LLM calls (default 3).

---

## Anti-patterns (when NOT to call these tools)

- **Don't call `calibrate` on every PR.** Use `evaluate` + `diff` for routine regression checks; reserve `calibrate` for when the prompt itself or the dataset structure changed.
- **Don't call `antemortem.run` on trivial changes.** Renaming a variable, fixing a typo, adjusting copy — the recon discipline is overhead in those cases.
- **Don't call `grade` with `strategy="llm"` in tight loops.** It's an LLM call per invocation. Use `strategy="rule"` or `strategy="ensemble"` for self-checks during agent reasoning.
- **Don't skip `preflight` when changing vendors.** Capability events recorded silently during a calibration will show up later as inexplicable regression; catch them up front.

---

## Composability

All four MCP servers consume Pydantic-modeled inputs/outputs and emit machine-readable artifacts. An agent can chain tools across servers without translation:

- `mini_antemortem_cli.analytical_preflight` → returns the same `AnalyticalFinding` shape as `omegaprompt.classify_traps`.
- `mini_omega_lock.empirical_preflight` → returns the same `JudgeQualityMeasurement` / `EndpointMeasurement` / `PerformanceMeasurement` shapes that `omegaprompt.preflight` consumes through the `omegaprompt.preflight.contracts` interface.
- `omegaprompt.evaluate(params=artifact)` accepts a `CalibrationArtifact` directly.

The toolkit is one substrate; the four servers are entry points to it.

---

## Cost-conscious orderings

If LLM budget is a hard constraint, run the cheap tools first and short-circuit on their results:

```
1. mini_antemortem_cli.analytical_preflight   (0 LLM calls)
2. omegaprompt.preflight                       (0 LLM calls; capability-only)
3. mini_omega_lock.compute_context_margin      (0 LLM calls)
4. omegaprompt.measure_sensitivity             (~10-30 calls)
5. mini_omega_lock.empirical_preflight         (~3-15 calls)
6. omegaprompt.calibrate                       (50-500 calls)
```

Stop at the first tool that returns a blocker. The first three steps are free; together they catch perhaps 60% of misconfigurations. The expense escalates monotonically downward.

---

**This document is the canonical agent-trigger reference for the toolkit.** When sub-tools' READMEs need a more detailed flow than they document inline, they point here. Last updated: 2026-04-29.
