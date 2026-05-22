# omegaprompt Trust Model

`omegaprompt` is an audit tool for prompt calibration. Its artifact is useful
because it records the test conditions, gates, provider capabilities, and risk
boundary crossings that produced a result. It is not proof that a prompt will
succeed in production.

## What the Artifact Proves

A `CalibrationArtifact` proves that a specific local run completed under the
recorded inputs:

- dataset paths or inline data resolved into `Dataset` records
- a declared `JudgeRubric`
- candidate `PromptVariants`
- selected target and judge providers
- execution profile (`guarded` or `expedition`)
- pre-declared walk-forward thresholds
- provider capability declarations and any degradation events
- final status and `ship_recommendation`

The artifact is machine-readable evidence for review and CI. It can prove
that a run passed its declared gates on the provided slices. It can also prove
that a run failed a gate, crossed a guarded boundary, or relied on a degraded
provider path.

## What the Artifact Does Not Prove

The artifact does not prove production success, global prompt superiority,
provider superiority, model superiority, or benchmark leadership. It cannot
prove that the dataset fully represents future traffic. It cannot prove that an
LLM judge will remain stable across provider model updates. It cannot turn an
expedition run into a ship-grade run.

Exact benchmark aggregate metrics and provider/model superiority claims are
disallowed unless they are tied to local reproducible artifacts, deterministic
reference artifacts, generated docs, or an explicit qualitative marker. The
default public reference path is `examples/reference/reproduce_reference_artifact.py`.

## Train/Test Split Discipline

Prompt calibration can overfit just like parameter calibration. `omegaprompt`
therefore treats the training slice and held-out slice as different roles:

- training data drives candidate selection
- held-out data checks whether the selected candidate still holds up
- thresholds are declared before scoring
- a failed holdout gate blocks ship advice

Changing the threshold after seeing scores invalidates the gate. The artifact
records the active `max_gap_threshold`, `min_kc4_threshold`, `validation_mode`,
and pass/fail result so a reviewer can see the rule that was actually applied.

## Walk-Forward and KC4 Semantics

Walk-forward validation compares train and test behavior for the selected
candidate. The gap check asks whether test fitness stayed close enough to train
fitness. KC4 is the holdout correlation check: when train and test slices share
enough item ids, per-item scores should track across slices rather than invert
or collapse.

`FAIL_KC4_GATE` is a ship blocker. A high calibrated fitness does not override
that status. `omegaprompt diff` also treats a candidate artifact with a failing
status or `hold`/`block` recommendation as a regression condition for Prompt CI.

KC4 is a guardrail, not a prophecy. A pass means the declared holdout check
passed on the provided slices. It does not prove production success.

## Paired, Disjoint, and Auto Validation

`validation_mode` records how to interpret the train/test relationship:

- `paired`: train and test share item ids by design; KC4 is expected and fails
  closed if the shared set is insufficient or structurally invalid.
- `disjoint`: train and test intentionally use different item ids; KC4 is not
  applicable and the gate is gap-only.
- `auto`: historical compatibility mode; compute KC4 when enough shared ids
  exist, otherwise record the explicit non-computed status.

Paired validation can measure correlation, but it may still miss production
coverage gaps. Disjoint validation can measure held-out generalization shape,
but it cannot compute holdout correlation. `auto` is convenient for migration,
but a production process should choose the mode intentionally.

## Holdout Correlation Limits

Holdout correlation is only as good as the held-out slice and score quality. It
is sensitive to tiny sample sizes, missing per-item scores, zero variance, and
judge drift. The artifact records `shared_item_count`, `kc4_status`, and
`kc4_correlation` so unmeasurable correlation is visible instead of silently
treated as success.

## Cross-Vendor Judge Discipline

LLM-as-judge can share biases with the target model. If target and judge use
the same vendor or model family, a response can score well because the judge
likes the same distributional habits as the target. `omegaprompt` treats target
and judge as separate provider call sites so teams can use cross-vendor judging
when they need a stronger independence signal.

Cross-vendor judging is still evidence, not truth. It reduces one structural
self-agreement risk; it does not prove that either provider is superior.

## Guarded vs Expedition Profiles

`guarded` is the default trust boundary. It blocks hidden fallbacks, weak judge
paths, placeholder providers, and structural risk that would make ship advice
misleading.

`expedition` allows explicit boundary crossings for exploration. It can record
schema fallback, experimental adapters, or relaxed safeguards, but the artifact
must make those crossings visible. Expedition is not ship-grade by default.

See [profiles-and-risk-boundaries.md](profiles-and-risk-boundaries.md) for the
operational boundary language.

## Provider Capability Degradation

Each provider declares `ProviderCapabilities`: strict schema support, JSON
object support, reasoning profile support, usage accounting, LLM judge
suitability, tier, experimental status, and placeholder status.

If a provider call degrades at runtime, the adapter emits a `CapabilityEvent`.
Those events propagate into `EvalItemResult`, `EvalResult`, and
`CalibrationArtifact`. A local endpoint falling back from strict schema to JSON
parsing is not hidden; it becomes audit data and may cross the guarded boundary.

See [provider-capabilities.md](provider-capabilities.md) for the current
provider matrix.

## Offline and Live Evidence

The offline deterministic path uses in-memory providers and judges:

```bash
python examples/reference/reproduce_reference_artifact.py
omegaprompt check-artifact examples/reference/reference_artifact.json --strict
```

This proves the local package, schema, reference artifact, integrity checker,
and report/diff paths work without network or API keys.

The live provider path uses real provider SDKs and API keys. Live runs are
local evidence for a specific provider/model/date/configuration. They are not
default CI evidence and must not be converted into public superiority claims
without a reproducible local artifact or generated document.

## Why Default CI Avoids Live Providers

Default CI must be deterministic, repeatable, and safe for contributors. Live
provider calls introduce network availability, account state, rate limits,
model drift, billing, and secret-management variability. Therefore default CI
uses mocks, deterministic reference artifacts, and contract tests. Optional
live tests live behind `OMEGAPROMPT_LIVE_PROVIDER_TESTS=1`.

Tooling or environment failures are classified as `TOOLING_MISSING` or
`ENVIRONMENT_BLOCKED`, never as release approval.

## MCP Optional-Extra Boundary

Core install is `omegaprompt`. MCP support is optional and installed as
`omegaprompt[mcp]`. Core imports and the core CLI must not require the MCP SDK.
If `omegaprompt-mcp` is invoked without the optional extra, the failure should
tell the user to install `omegaprompt[mcp]` and classify the problem as
`TOOLING_MISSING`.

The MCP server exposes the same runtime entrypoint semantics as Python and CLI
wrappers; it does not create a second calibration policy.

## Prompt CI Diff Usage

Use `omegaprompt diff baseline.json candidate.json` in CI after producing a new
artifact. A diff regression can be caused by lower calibrated fitness, lower
walk-forward test fitness, hard-gate regression, cost/latency frontier
regression, failed status, `hold`/`block` ship recommendation, or a new guarded
boundary crossing.

The diff result is not a benchmark leaderboard. It is a local regression gate
between two artifacts under the team's declared contract.
