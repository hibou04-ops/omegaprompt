# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.1.1] - 2026-06-12

Marketplace: shorten action.yml description to GitHub's 125-char limit so the
composite Action can be published to GitHub Marketplace. No library/API changes.

### Changed

- Shortened the composite `action.yml` `description` to a single line under
  GitHub's 125-character Marketplace limit (107 chars) so the ship gate can be
  published to the GitHub Marketplace and consumed as
  `uses: hibou04-ops/omegaprompt@v2.1.1`. The action's inputs, outputs, run
  steps, and branding are unchanged; the example workflow pin
  (`examples/ci/ship-gate.yml`) advanced to `2.1.1`.

### Notes

- No source, behavior, or public-API change. `CalibrationArtifact.schema_version`
  remains `2.0`, the MCP tool set stays frozen at 8 (no new tool), the CLI command
  set is unchanged, and the omega-lock dependency pin remains `>=0.3.0,<0.4.0`.
  Fully backward compatible; `publish.yml` reads the version from `pyproject.toml`.

## [2.1.0] - 2026-06-12

### Added

- **`omegaprompt gate` (new CLI command).** A first-class, zero-network CI ship gate. It fuses the artifact integrity audit (the same checks `check-artifact` runs) with the train↔holdout transfer/gap (overfit) verdict, then exits `0` (clear to ship), `1` (ship-blocked), or `2` (environment/load failure). `--format json` emits a schema-versioned machine summary (`gate_schema_version` `1.0`). Shipping is now a dedicated decision rather than something inferred from `diff`/`report`. Exposed in Python as `omegaprompt.run_gate()` / `GateResult`. (No MCP tool was added: the MCP tool set is frozen at 8 and the consistency contract enforces it — see Notes.)
- **`ollama` provider adapter (distinct named class).** `make_provider("ollama")` now returns a clearly-named `OllamaProvider` with Ollama defaults (local `http://localhost:11434/v1` base URL, keyless) instead of a generic `local` adapter configured via a `backend` string. Backward-compatible: same name, same OpenAI-compatible transport, same LOCAL/experimental capability reporting. Supported providers are now `anthropic / openai / gemini / local / ollama / vllm / llama_cpp`.
- **HTML report.** `omegaprompt report --format html` renders a self-contained, single-file scorecard (inline CSS, no JS, no external assets — stdlib only) with status, overfit verdict, summary metrics, sensitivity ranking, and parameters.
- **`--format json` on `report` and `diff`.** `report` gains a stable, schema-versioned summary projection (`summary_schema_version` `1.0`) that includes the prominent overfit block; `diff` now exposes its existing structured `ArtifactDiff` as a deterministic `--format json` output. Both are sorted-key, byte-stable for CI diffs.
- **Overfit metrics surfacing.** `extract_overfit_metrics()` / `OverfitMetrics` / `overfit_metrics_dict()` surface the train↔holdout *transfer correlation* (KC-4) and *generalization gap* — the "is my prompt overfit?" numbers — as one prominent machine-readable block, with a coarse `overfit_verdict` (`GENERALIZES` / `OVERFIT` / `UNVERIFIABLE` / `UNKNOWN`). This is a **pure read** over the existing walk-forward block: it adds **no** field to `CalibrationArtifact`, so the artifact schema stays `2.0` and every golden hash is byte-stable.
- **GitHub composite Action.** `action.yml` at the repo root wraps `omegaprompt gate`, so downstream repos can `uses: hibou04-ops/omegaprompt@v2.1.0` in CI. Inputs: `artifact`, `format`, `require-generalization`, `python-version`, `version`. Outputs: `passed`, `exit-code`. An example workflow lives at `examples/ci/ship-gate.yml`.
- **Determinism/replay hardening.** The offline golden-reference replay now also asserts the new gate JSON and report-summary JSON are byte-stable across repeats and across a save→load roundtrip, so CI catches any nondeterminism in the new surfaces without the network.

### Changed

- **`publish.yml` is version-agnostic.** The "Verify version" step now reads the single source of truth (`[project].version` in `pyproject.toml` via `tomllib`), asserts `src/omegaprompt/__init__.__version__` matches, and — when invoked from a release — asserts the tag equals `v<version>`. No more hardcoded version string in the workflow.
- **Distribution-focused README family overhaul.** `README.md` is rebuilt around a search-optimized, jargon-free front page (angle: *"the overfit gate for your prompts"*) that leads with the new `omegaprompt gate` CI hero and the `uses: hibou04-ops/omegaprompt@v2.1.0` Action, a comparison table (omegaprompt vs promptfoo/DSPy vs eyeballing the eval), the honest offline demo (noting the bundled split degrades the per-item transfer gate to gap-only — `MISSING_PER_ITEM_SCORES`), and the machine-readable overfit-metrics block; the full reference body follows below it. Internal engine jargon was scrubbed from all README-family prose: *KC-4 → held-out transfer gate*, *guarded/expedition → strict mode / fast mode*, *walk-forward → held-out validation* (frozen code symbols like `run_p1` / `P1Config` / `walk-forward` remain only inside code blocks with a one-line gloss). `EASY_README.md` was rewritten plain story-first. `README_KR.md` and `EASY_README_KR.md` are faithful Korean translations (code/tables identical, prose translated). All four cross-link each other.
- **Dynamic README badges.** The PyPI version, Python, and License badges are now live `shields.io` endpoints (`pypi/v`, `pypi/pyversions`, `pypi/l`) with `?cacheSeconds=3600`, so they read the published version from PyPI and never go stale on a release bump (replacing the static `pypi-X.Y.Z` shield). The 8-badge composition and order are unchanged; the consistency checker's `README_BADGES` tokens and the `README_PYPI_BADGE_VERSION` / release-audit version-alignment checks were updated to recognize the dynamic PyPI badge (version owned by PyPI, cannot drift).

### Notes

- Package version is `2.1.0`; `CalibrationArtifact.schema_version` remains `2.0`. The omega-lock dependency pin remains `>=0.3.0,<0.4.0`.
- All additions are backward-compatible: the 68 `omegaprompt.__init__` exports are additive-only (new symbols added, none removed/renamed); the frozen downstream contract surfaces for `mini-omega-lock` / `mini-antemortem-cli` are untouched; console scripts, MCP tools (frozen at 8 — `gate` is CLI/Python only), runtime entrypoints (8), and the artifact schema (`2.0`) are unchanged. The consistency checker's `EXPECTED.cli_commands` was deliberately extended with `gate`.
- Golden reference artifacts are **unchanged** — overfit surfacing reads the artifact rather than mutating it, so the golden hashes are byte-stable.
- This release-preparation change edits the working tree only; it does not publish to PyPI, push tags, or create/edit GitHub Releases.

## [2.0.2] - 2026-06-08

### Added

- **Opt-in parallel item evaluation.** `CalibrateTuning.max_workers` (Python) / `--concurrency` (CLI) parallelize per-item target+judge evaluation within each candidate via a local `concurrent.futures.ThreadPoolExecutor`. Each item still runs its target call then its judge call sequentially, so the number of concurrent calls to any one provider never exceeds the configured value. Default `1` (serial) produces byte-identical artifacts to prior versions. Wall-clock speedup is bounded by — and conditional on — your provider account's concurrency ceiling (RPM/TPM): roughly ~50% at `2`, ~75% at `4`, ~85% at `8` when the account permits it, and **none** if the account effectively serializes calls. No cross-provider rate-limit handling; configure per your account (a value above your real ceiling produces 429s, not speedups).

### Changed

- **Best-candidate evaluation is memoized.** `PromptTarget.evaluate()` now caches results keyed on the resolved params, so the final best-candidate evaluation reuses the grid-search result instead of re-calling the providers. This reduces live API calls during `calibrate()`. No public-surface change (`EvalResult` / `CalibrationArtifact` / `CalibrableTarget` unchanged) and no artifact-schema change.

### Notes

- Package version is `2.0.2`; `CalibrationArtifact.schema_version` remains `2.0`. The omega-lock dependency pin remains `>=0.3.0,<0.4.0`.
- Golden reference artifacts are **unchanged** — the deterministic harness uses fixed counts/latency and builds artifacts directly (not via `runtime.calibrate()`), so neither feature affects the golden hash.
- This release-preparation change edits the working tree only; it does not publish to PyPI, push tags, or create/edit GitHub Releases.

## [2.0.1] - 2026-06-07

### Fixed

- **omega-lock 0.3.0 compatibility.** omega-lock 0.3.0 renamed the target-result action count `n_trials` -> `sample_count` (KC-3 action floor) and reads `.sample_count` off the result `PromptTarget.evaluate` returns (in `stress.py`, `grid.py`, and `walk_forward.py`). `EvalResult` now exposes `sample_count` as a read-only property aliasing the stored `n_trials` field, so the calibration path no longer crashes with an `AttributeError` against the 0.3.0 seam. The stored field stays `n_trials`, so serialized artifacts and golden reference hashes are unchanged.

### Added

- **Consumer docking hardlock.** A fail-loud contract test (`tests/test_omega_lock_contract.py`) asserts the omega-lock dependency seam — the specific fields and call signatures omegaprompt consumes off omega-lock's result types — so a future producer-side rename fails LOUD (named, located, pre-merge) instead of surfacing as a silent mid-run `AttributeError`. A scheduled consumer canary workflow (`.github/workflows/omega-lock-compat.yml`) runs omegaprompt's own consumer contract and integration tests against `omega-lock@main`, ahead of the pinned release, so a producer rename of any consumed field is caught before omegaprompt bumps its pin. Isolated from the PR gate on purpose: it triggers only on a schedule / manual dispatch.

### Notes

- Package version is `2.0.1`; `CalibrationArtifact.schema_version` remains `2.0`. The omega-lock dependency pin remains `>=0.3.0,<0.4.0`.
- This release-preparation change edits the working tree only; it does not publish to PyPI, push tags, or create/edit GitHub Releases.

## [2.0.0] - 2026-05-23

### Added

- Trust-first release gates for repository consistency, generated README claims, artifact integrity, deterministic golden reference artifacts, provider/MCP contracts, wheel smoke, release audit, publish readiness, and post-release verification.
- Public claim ledger and generated README claim document so README/PyPI-facing claims are tied to source-of-truth, generated-doc, reproducible-command, deterministic-artifact, or qualitative-marker evidence.
- Measurement-grade offline golden reference harness with multiple deterministic artifacts, a golden manifest, normalized artifact hashes, and drift detection.
- Local release and collaboration tooling: release checklist, scope freeze, release draft generator, issue/PR templates, and post-release verifier with explicit local-only and network modes.
- Markdown link checker for README/PyPI-safe file links, anchors, and case-sensitive repository paths, wired for no-network CI use.

### Changed

- README family now separates deterministic offline evidence from opt-in live provider paths and clarifies the `omegaprompt`, `omegacal`, `omega-lock`, MCP extra, and CLI name boundaries.
- Provider capability documentation and tests now make Gemini/local adapter status measurable without default live API calls.
- Default CI remains no-network for providers and checks generated claims, reference artifacts, artifact integrity, provider/MCP contracts, and wheel smoke.

### Notes

- Package version is `2.0.0`; `CalibrationArtifact.schema_version` remains `2.0`.
- This release-preparation commit does not publish to PyPI, push tags, or create/edit GitHub Releases.

## [1.7.4] - 2026-05-07

### Fixed

- **Reverted PyPI badge to hardcoded `pypi-1.7.4-blue.svg`.** Dynamic shields endpoint (`pypi/v/omegaprompt.svg`) was rendering stale on PyPI pages for >24 hours. Manual bump per release is the price for visible accuracy. Each release MUST bump this badge in lockstep with `pyproject.toml` version.

## [1.7.3] - 2026-05-07

### Fixed

- **Test count badge synced with reality** — was 227 (stale), actual is 317. README + README_KR updated.
- No code changes.

## [1.7.2] - 2026-05-07

Re-publish to refresh the PyPI page rendering. No code changes.

## [1.7.1] - 2026-05-07

### Fixed

- **PyPI badge now dynamic.** Switched README + README_KR PyPI version badge from a hardcoded `pypi-X.Y.Z-blue.svg` shield to the dynamic `pypi/v/omegaprompt.svg` endpoint. Previously every release required manual badge bump (and one was missed in 1.7.0, leaving the page rendering with `pypi 1.6.0` next to the 1.7.0 release header). The dynamic endpoint queries PyPI for the latest version, so future releases never drift.

## [1.7.0] - 2026-05-07

Docs follow-up to 1.6.0. No code changes; version bumped to minor for visibility on the PyPI page (the 1.6.0 page rendered with a stale README and pre-Gemini badge).

### Changed

- README badge bumped 1.4.0 → 1.6.0 (was stale on PyPI page despite the 1.6.0 release).
- Top-of-README v1.6.0 callout under `pip install` summarising the Gemini 2.5 Flash adapter.
- Quick start `GEMINI_API_KEY` line gains free-tier link to <https://aistudio.google.com/apikey>.
- New **Troubleshooting** section: per-provider env-var matrix (no cross-vendor fallback), missing-key error, deterministic smoke test, single-call provider sanity, Gemini guarded-judge refusal.
- **`README_KR.md` synced** with the 1.6.0 / 1.6.1 changes (Gemini callout, env-var matrix, Troubleshooting in Korean).

## [1.6.0] - 2026-05-06

### Added

- **Real Gemini 2.5 Flash provider** (replaces the prior placeholder). Uses the official Google GenAI SDK (`google-genai`) with freeform / JSON-object / strict-schema (`response_schema`) request paths plus local Pydantic validation. Default model: `gemini-2.5-flash`. Live single-call smoke test verified against `gemini-2.5-flash` (input_tokens=21, output_tokens=2 normalized correctly).
- **Capability declaration**: `tier=cloud`, `supports_strict_schema=True`, `supports_json_object=True`, `supports_llm_judge=True`, `ship_grade_judge=False` — guarded judge use requires separate validation; expedition profile may fall back to JSON-object output + local validation with a `CapabilityEvent`.
- **Env key resolution**: `_ENV_KEYS_FOR_PROVIDER` accepts `GEMINI_API_KEY` or `GOOGLE_API_KEY` (tuple form, per-provider).
- **`google-genai>=1.0.0` dependency** added to base `dependencies`; new `gemini` extra for slim installs.

### Changed

- `normalize_usage` handles Gemini's `prompt_token_count` / `candidates_token_count` / `total_token_count` fields in addition to OpenAI/Anthropic field names.
- README provider matrix, capability table, and sample commands updated to describe the real adapter.
- `LLMProvider` Protocol unchanged — Anthropic/OpenAI/local adapters are not modified in this release.

### Test

- 41 provider tests pass (mock-based for all three vendors).
- Live verification: Anthropic ✅ (claude-opus-4-7), Gemini ✅ (gemini-2.5-flash). OpenAI not verified due to user-side API key issue (provider code unchanged from 1.5.0).

### Notes

- Reasoning profiles (`LIGHT`/`DEEP`) emit a `CapabilityEvent` for Gemini because this adapter does not map them to a native Gemini control. `OFF` / `STANDARD` work without events.
- Gemini judge is not marked `ship_grade_judge=True`. Validate independently before treating Gemini-judged artifacts as ship-ready under guarded mode.

## [1.4.0] - 2026-04-29

IP defense package + cross-toolkit AGENT_TRIGGERS guide. Same eight runtime entrypoints + MCP server as 1.3.0; this release ships the previously-out-of-tree authorship and trigger artefacts inside the PyPI sdist for parity with the rest of the toolkit and adds the canonical agent-trigger cookbook.

### Added

- **IP defense package shipped in sdist**: `NOTICE` (with explicit Kyunghoon Gwak / 곽경훈 author binding), `AUTHORS.md`, `PRE_EXISTING_IP.md`, `IP_DEFENSE_CHECKLIST.md`. Same pattern as omega-lock / Antemortem / antemortem-cli / mini-omega-lock / mini-antemortem-cli — six-repo toolkit now consistent.
- **`AGENT_TRIGGERS.md`** at repo root — canonical *when-to-call-which-tool* cookbook spanning the four MCP servers in the toolkit (omegaprompt, antemortem, mini-omega-lock, mini-antemortem-cli; 18 tools total). Maps nine agent scenarios (pre-implementation reconnaissance, pre-calibration sanity, pre-ship validation, PR regression check, production canary, agent self-grading, "is calibrate worth it?", performance projection, new vendor onboarding) to recommended tool sequences with cost-conscious orderings. README's MCP section links to it.
- **Sdist `include` extended** to ship `NOTICE`, `AUTHORS.md`, `PRE_EXISTING_IP.md` (matching the discipline applied across the rest of the toolkit during the 2026-04-29 IP-package consolidation).

### Fixed

- **Author email attribution clarified.** Earlier docs claimed `hibou04@gmail.com` was a verified personal account of the Primary Author; this was incorrect. `hibouaile04@gmail.com` is the only verified personal email; `hibou04@gmail.com` was an unintended local git client misconfiguration that put a non-author email in some commit `author` fields prior to 2026-04-29. AUTHORS.md and PRE_EXISTING_IP.md now state this explicitly. Author *name* `Hibou04-ops` remains the unambiguous identifier across the repository's history. From 2026-04-29 onwards every repo's local git config commits as `hibouaile04@gmail.com`.

### Rationale

1.3.0 shipped the eight runtime entrypoints + MCP server but was tagged before the toolkit-wide IP-defense alignment landed and before the cross-server agent-trigger cookbook was written. 1.4.0 closes that gap so the GitHub HEAD and the PyPI sdist contain the same artefacts. No code-path changes from 1.3.0 — same runtime, same MCP tools, same test suite (181 tests passing).

## [1.3.0] - 2026-04-29

Eight one-call runtime entrypoints + MCP server. The calibration kernel is unchanged; the change is the *agent-callable surface* layered on top of it. The same eight operations are now reachable from Python (one call each), the CLI (existing surface, preserved), and an MCP server that Claude Code / Cursor can spawn over stdio.

### Added

- **`omegaprompt.runtime` module** with eight high-level entrypoints, re-exported at package level (`from omegaprompt import calibrate, evaluate, ...`):
  - **Tier 1** (most-used): `calibrate` (full pipeline), `evaluate` (single-config eval, no search), `report` (artifact → markdown), `diff` (compare two artifacts).
  - **Tier 2** (less-frequent but distinct): `measure_sensitivity` (cheap axis-stress probe, no grid), `grade` (score one response, rule / llm / ensemble strategy), `preflight` (capability-only env sanity check), `classify_traps` (deterministic trap classification, requires `mini-antemortem-cli`).
  - Each entrypoint accepts paths or in-memory objects. Provider arguments accept `str | ProviderSpec | LLMProvider`. `evaluate(params=...)` accepts a `CalibrationArtifact` directly and auto-extracts `calibrated_params`.
- **Five new Pydantic types** for the agent-callable contract: `ProviderSpec`, `CalibrateTuning`, `SensitivityTuning`, `SensitivityResult`, `ArtifactDiff`. All declare `extra="forbid"` so MCP clients receive strict JSON schemas.
- **`omegaprompt.mcp` package** — FastMCP server exposing all eight runtime entrypoints as agent-callable tools. Run with `python -m omegaprompt.mcp` (stdio, default for Claude Code) or `--http` (streamable-http). Console script `omegaprompt-mcp` registered.
- **New optional dependency `[mcp]`** — `pip install "omegaprompt[mcp]"` pulls in the official MCP Python SDK (`mcp>=1.0.0`).
- **35 new tests** — `test_runtime.py` (26 tests covering Tier 1 + Tier 2 pure functions and type coercion) and `test_mcp_server.py` (6 smoke tests verifying all eight tools register with input schemas and required-args contracts that match the runtime signatures). Total **181 tests passing**, zero network calls in CI.

### Changed

- **README section 10 (Quick start)** restructured into three subsections (10.1 Python high-level API, 10.2 CLI, 10.3 MCP server) so all three calling surfaces are visible to first-time readers.
- **Abstract** updated to flag the eight runtime entrypoints and the MCP-substrate framing.

### Rationale

Up through 1.2.0, omegaprompt's agent-friendly surface was the CLI. That works for human-driven scripts and CI gates but forces an agent to either spawn a subprocess or compose six layers of the underlying API to do anything. 1.3.0 closes that gap: the same calibration kernel is now reachable through one Python call, one CLI command, or one MCP tool invocation. Sensitivity-driven input design (HIGH-frequency knobs flat, LOW-frequency grouped under `*Tuning` escape hatches) keeps the agent-facing surface small without giving up power-user control.

The MCP server is the framing change. omegaprompt is positioned as the *substrate* agent runtimes plug into for prompt validation — calibrate-before-ship, regress-on-PR, classify-traps-before-acting — rather than a manual tool that a human has to remember to run.

## [1.2.0] - 2026-04-27

PyPI metadata alignment + 9-task calibration suite + 60s demo. The package itself was already Apache 2.0 in `LICENSE` from earlier in the week; this release synchronises the PyPI license classifier, ships the validated calibration suite, and pins a verbatim 60-second walkthrough that anyone can replay.

### Added

- **9-task prompt calibration suite** (`examples/`) covering `code_review`, `code_writing`, `debugging`, `refactoring`, `explanation`, `summarization`, `translation`, `test_writing`, `commit_message`. Each task ships:
  - `variants.json` (5 prompt variants per task: baseline → role+terse → format → rules+format → guard)
  - `train.jsonl` (8-16 representative items with reference solutions)
  - `rubric.json` (3 dimensions + 3 hard_gates)
  - `_subagent_outputs/` and `claude_results.json` (Opus 4.7 simulation outputs + tokenised comparison + winner)
  - `quality_review.json` (independent Sonnet scoring on 8 of 9 tasks)
- **Validated winners with measured token savings vs verbose baseline** (token-only ranking on Opus 4.7 subagent simulation):
  - `code_review`: V5 NO_ISSUE guard, **−57.5%**, 16/16 perfect quality
  - `code_writing`: V3' rules+format, 8/8 correct
  - `debugging`: V4 INSUFFICIENT_INFO guard, 8/8 correct
  - `refactoring`: V4 ALREADY_CLEAN guard, **−65.6%**
  - `explanation`: V4 ALREADY_OBVIOUS guard
  - `summarization`: V4 ALREADY_BRIEF guard
  - `translation`: V4 NO_TRANSLATION_NEEDED guard
  - `test_writing`: V4 PARAMETRIZE
  - `commit_message`: V2 conventional commits, **−52.3%**
- **5 winner reassignments** after independent Sonnet quality reviews overrode pure-token ranking on `code_writing`, `debugging`, `explanation`, `summarization`, `test_writing`. Rationale: token-only winners had hard-gate violations or coverage gaps that quality scoring caught.
- **60-second demo walkthrough** (`examples/demo_calibration.py`, `examples/demo_replay.py`, `docs/demo/omegaprompt-demo.en.srt`). Reproducible via `PYTHONIOENCODING=utf-8 python examples/demo_replay.py`. README embeds the inline video.
- **Reusable backtest CLI** (`backtest.py`) usable with Ollama + tiktoken for offline calibration validation.
- **Cross-model 7B validation** for `code_review` (exaone3.5:7.8b, gemma4:e4b) showing 7B gives directional signal but unreliable absolute quality numbers.

### Fixed

- **PyPI license classifier**: 1.1.0 was published with `License :: OSI Approved :: MIT License` because the relicense to Apache 2.0 happened after the 1.1.0 PyPI upload. 1.2.0 publishes with the correct `License :: OSI Approved :: Apache Software License` classifier, matching the `LICENSE` file already in the repository.

### Methodology Note

Token-only ranking turned out wrong on 5 of 9 tasks once independent Sonnet quality scoring ran — V4 USE_STDLIB missed `@wraps` in `cw6`, V2 debugging format failed `db8` fix_executable, V2 explanation lacked guard, V1 summarization lost to V4 by 0.03 quality but V4 fires the brevity guard correctly, V3' test_writing had two coverage gaps. **Quality scoring is mandatory for production deployment decisions** — see `examples/code_review/validation_archive.json` for the full 8-backtest + 5-quality-review audit trail on the most validated task.

## [1.0.0] - 2026-04-22

v1.0 is a provider-neutral redesign. The calibration discipline (sensitivity-driven coordinate descent, `hard_gate × soft_score` fitness, walk-forward ship gate with KC-4 Pearson threshold, machine-readable artifacts) is preserved intact; the *contract* around it is restructured so omegaprompt is no longer implicitly Claude-shaped.

### Added

- **Provider-neutral meta-axes** (`omegaprompt.domain.enums`) replace Claude-specific axes. The calibration searcher now probes `reasoning_profile` (off/light/standard/deep), `output_budget` (small/medium/large), `response_schema_mode` (freeform/json_object/strict_schema), and `tool_policy` (no_tools/tool_optional/tool_required). Each provider adapter maps these to its vendor's native parameters internally; the public contract carries no vendor knobs.
- **Unified `LLMProvider.call(ProviderRequest) -> ProviderResponse`** — the two-method (`complete` / `structured_complete`) interface from v0.2 collapses into a single boundary. STRICT_SCHEMA mode dispatches to the vendor's native parse path (`messages.parse` on Anthropic, `beta.chat.completions.parse` on OpenAI); FREEFORM and JSON_OBJECT dispatch to chat completion.
- **`Judge` protocol + three implementations** (`omegaprompt.judges`):
  - **`LLMJudge`** — rubric-based LLM grader using any provider's STRICT_SCHEMA path.
  - **`RuleJudge`** — deterministic gate checker with shipped helpers (`default_no_refusal`, `default_non_empty`, `json_object_check`, `regex_check`). Zero API cost; reproducible.
  - **`EnsembleJudge`** — runs `RuleJudge` first, escalates to a fallback judge only when rule gates pass. Cuts LLM-judge cost on obviously-broken responses.
- **Layered package structure** (`domain/`, `core/`, `providers/`, `targets/`, `judges/`, `commands/`). The kernel depends only on `domain` types; providers and judges plug in as adapters. Backward-compat shims in `omegaprompt.dataset` / `omegaprompt.judge` / `omegaprompt.fitness` / `omegaprompt.schema` / `omegaprompt.target` / `omegaprompt.api` / `omegaprompt.prompts` keep v0.2 import paths resolving during migration.
- **`MetaAxisSpace`** replaces `PromptSpace`. Declares axis bounds as enum lists rather than integer indices, so a CalibrationArtifact's `best_params` is self-describing ("reasoning_profile: deep" vs "effort_idx: 2").
- **Richer `CalibrationArtifact` schema** — adds `schema_version`, `sensitivity_ranking`, explicit `walk_forward` block (train/test/gap/KC-4/passed), `status` (OK / FAIL_KC4_GATE / FAIL_HARD_GATES / FAIL_NO_CANDIDATES), and `rationale`. All v0.2 `CalibrationOutcome` fields are preserved under aliases in the shim.
- **New CLI subcommands**:
  - `omegaprompt report <artifact>` — render a CalibrationArtifact as Markdown (for PR descriptions, CI step outputs).
  - `omegaprompt diff <old> <new>` — compare two artifacts, exit 1 on regression. Intended for CI.
- **New CLI flags on `calibrate`**: `--max-gap`, `--min-kc4` (walk-forward gate thresholds, both declared up front — cannot be lowered post-hoc).
- **~37 new tests** covering meta-axis enums, rule / LLM / ensemble judges, walk-forward Pearson gate, sensitivity ranking + top-K unlock, artifact round-trip, and the new CLI subcommands. Total **110 tests**, still zero network calls in CI.

### Changed

- **`omegaprompt.providers.base.LLMProvider`** is a fresh Protocol with a single `call()` method. The v0.2 `complete` / `structured_complete` methods are gone.
- **`PromptTarget` signature** now takes a `judge: Judge` rather than a `judge_provider: LLMProvider`. The previous behaviour (hard-coded LLM-judge call) is recovered as `judge=LLMJudge(provider=...)`; users who want rule-only scoring pass a `RuleJudge` directly; users who want cost optimisation pass an `EnsembleJudge`.
- **Package version and `description`** bumped; the project identity in `pyproject.toml` explicitly names "model-agnostic calibration engine" and enumerates all four provider-neutral meta-axes.
- **Judge system prompt** moves to `omegaprompt.judges.llm_judge.JUDGE_SYSTEM_PROMPT`. The v0.2 location `omegaprompt.prompts.JUDGE_SYSTEM_PROMPT` re-exports from the new path.

### Rationale

v0.2 shipped provider-pluggability at the SDK boundary but kept the calibration axes (`effort_idx`, `thinking_enabled`, `max_tokens_bucket`) named after Anthropic's API contract. The result: a user bringing a non-Claude target had to reason about Claude parameter names that their provider ignored. v1.0 finishes the provider-neutral work by renaming the *axes themselves* to provider-neutral categories and pushing vendor-specific translation into each adapter. An artifact now says `"reasoning_profile": "deep"`, not `"effort_idx": 2` — self-describing regardless of which vendor produced it.

The **three-judge split** (rule / LLM / ensemble) addresses two real costs: (1) LLM-judge cost dominates calibration bills once datasets exceed ~50 items, and (2) LLM judges drift across releases. Rule-based gate checks give reproducible format / refusal detection with zero API cost, and the ensemble pattern recovers the LLM judge's qualitative scoring only when the response clears the structural bar.

**Walk-forward gate strengthening** — v1.0 separates the `max_gap` and `min_kc4` thresholds into explicit CLI flags, and the artifact records the exact thresholds in force at ship time. CI pipelines reviewing an artifact can verify both the numbers and the pre-declared gates they were measured against.

### Migration from v0.2.x

Most v0.2 import paths continue to resolve via backward-compat shims, but the **semantics of the types they name changed**:

- `ParamVariants` is aliased to `PromptVariants` (same shape, new canonical name).
- `PromptSpace` is aliased to `MetaAxisSpace` **but the field set is different**: `effort_min/max`, `allow_thinking`, `max_tokens_bucket_min/max` are gone; `reasoning_profiles`, `output_budgets`, `response_schema_modes`, `tool_policy_variants` replace them. Any v0.2 code that constructed a `PromptSpace(effort_min=0, effort_max=2, allow_thinking=True, ...)` will need updating to `MetaAxisSpace(system_prompt_idx_max=N, reasoning_profiles=[...], output_budgets=[...])`.
- `CalibrationOutcome` is aliased to `CalibrationArtifact`. The v0.2 field set is a subset of v1.0's; v0.2 artifacts deserialise cleanly.
- `call_judge` and `_build_judge_payload` no longer exist. Port to `judge.score(rubric=..., item=..., target_response=...)` with an `LLMJudge` instance.
- `PromptTarget(target_provider=..., judge_provider=...)` -> `PromptTarget(target_provider=..., judge=LLMJudge(provider=judge_provider))`.

The JSON artifact format is forward-compatible: v0.2 artifacts validate against the v1.0 `CalibrationArtifact` schema (new fields default to empty/null).

## [0.2.0] - 2026-04-22

### Added

- **`LLMProvider` Protocol + `providers/` package** — calibration is now model-agnostic at both boundaries. Target and judge each speak to their own provider; they can be the same instance (common case) or different instances (cross-vendor validation). The discipline (stress + grid + walk-forward + KC-4) is vendor-neutral; only the two API-call boundaries are pluggable.
- **Anthropic adapter** (`omegaprompt/providers/anthropic_provider.py`) — `complete` via `messages.create` with `cache_control={"type": "ephemeral"}`, adaptive thinking + `effort`; `structured_complete` via `messages.parse(output_format=JudgeResult)`.
- **OpenAI adapter** (`omegaprompt/providers/openai_provider.py`) — `complete` via `chat.completions.create`; `structured_complete` via `beta.chat.completions.parse(response_format=JudgeResult)`. Accepts `base_url`, so Azure OpenAI, Groq, Together.ai, OpenRouter, and local Ollama all work as drop-ins.
- **`make_provider(name, model, api_key, base_url)`** factory + `supported_providers()` helper.
- **CLI flags on `omegaprompt calibrate`**: `--target-provider`, `--target-model`, `--target-base-url`, `--judge-provider`, `--judge-model`, `--judge-base-url`. Target and judge are independent — use different providers for cross-vendor validation (e.g. `--target-provider openai --judge-provider anthropic`).
- **Provider metadata in `CalibrationOutcome`**: `target_provider`, `target_model`, `judge_provider`, `judge_model` are now first-class fields on the artifact. Every calibration run records which models produced its numbers.
- **~15 new tests** covering the factory, both adapters (`complete` + `structured_complete` each), cross-SDK usage normalization, and cross-vendor target/judge combinations.

### Changed

- **Dependencies** now include both `anthropic>=0.40.0` and `openai>=1.50.0` by default. Slim installs via extras (`pip install omegaprompt[anthropic]` / `pip install omegaprompt[openai]`) available for users stripping one vendor.
- **`api.py` no longer imports the Anthropic SDK.** `call_target` and `call_judge` delegate to the configured `LLMProvider`.
- **`target.py` accepts two separate `LLMProvider` instances** (`target_provider`, `judge_provider`) instead of two clients. The provider names + models flow through to the `CalibrationOutcome` artifact so run reproducibility is machine-readable.
- **CLI output** surfaces the resolved target + judge providers and models on every run:
  `Target: openai/gpt-4o   Judge: anthropic/claude-opus-4-7`

### Rationale

v0.1.0 pinned a single Claude model as both target and judge. That was coherent for a tight vendor-first prompt contract but the wrong framing for a calibration discipline. The guarantees omegaprompt provides (Pydantic-enforced judge output, hard gates collapsing fitness to zero, KC-4 walk-forward gate) are not Claude-specific; they are ML-overfitting defenses ported to the prompt setting. v0.2 reframes the tool to match: provider-pluggable at both boundaries, with each adapter using its vendor's strongest native schema-enforcement mechanism so precision is preserved per provider.

**Cross-vendor validation** is a genuine capability gain. Judging OpenAI outputs with an Anthropic judge (or vice-versa) gives a stronger-than-peer signal than same-vendor self-judging. Local models via Ollama (`--target-base-url http://localhost:11434/v1`) allow zero-API-cost calibration of open-weight prompts.

### Migration from v0.1.x

Existing callers with `ANTHROPIC_API_KEY` set see no behavior change — both target and judge default to Anthropic. Users who want OpenAI or a compatible endpoint add `--target-provider openai` and/or `--judge-provider openai`. The `CalibrationOutcome` JSON schema gained four optional fields (all default `null`); existing outcome files still validate.

## [0.1.0] - 2026-04-22

Initial public release. Ports omega-lock's calibration methodology - stress-based sensitivity analysis, top-K unlock grid search, and KC-4 walk-forward validation - to the prompt engineering setting.

### Added

- `PromptTarget` - implements omega-lock's `CalibrableTarget` protocol. Exposes five calibratable axes.
- `JudgeRubric` + `Dimension` + `HardGate` - user-supplied LLM-as-judge rubric.
- `JudgeResult` - Pydantic-enforced judge response.
- `CompositeFitness` - `hard_gate × soft_score` aggregation.
- `Dataset` / `DatasetItem` - JSONL loader.
- `ParamVariants` / `PromptSpace` - configuration for the search axes.
- `omegaprompt calibrate` CLI.
- Prompt caching on the judge system prompt.

### Scope boundary - not in v0.1

- Multi-judge validation, HTML report rendering, benchmark harness, proprietary hosting.

### Rationale

omega-lock's insight — separate `"found something"` from `"it generalizes under constraints"` — applies verbatim to prompt engineering. The port reused ~80% of the omega-lock calibration engine; only the `PromptTarget` adapter, the `JudgeRubric` + `JudgeResult` types, and a thin CLI are new.

[Unreleased]: https://github.com/hibou04-ops/omegaprompt/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/hibou04-ops/omegaprompt/releases/tag/v1.0.0
[0.2.0]: https://github.com/hibou04-ops/omegaprompt/releases/tag/v0.2.0
[0.1.0]: https://github.com/hibou04-ops/omegaprompt/releases/tag/v0.1.0
