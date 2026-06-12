# omegaprompt 2.1.0 — ship gate, output formats, overfit surfacing

This release is **additive and backward-compatible**. The artifact schema
stays `2.0`, the 68 `omegaprompt.__init__` exports are additive-only, and
the frozen downstream contract surfaces are untouched. Nothing here changes
how existing calibration runs behave.

## `omegaprompt gate` — the dedicated CI ship gate

Before 2.1.0, shipping was inferred from `diff` / `report`. `gate` makes it
a first-class, zero-network decision. It fuses two existing checks:

1. **Artifact integrity** — the same audit `check-artifact` runs (schema +
   semantic validity, walk-forward shape, provider-capability coherence,
   canonical roundtrip).
2. **Holdout transfer / gap verdict** — the train↔holdout generalization
   read: did the candidate clear its pre-declared walk-forward gate, or did
   it overfit the training slice?

```bash
omegaprompt gate artifact.json                 # human-readable, exit 0/1/2
omegaprompt gate artifact.json --format json   # schema-versioned summary
omegaprompt gate artifact.json --no-require-generalization
```

Exit codes:

| code | meaning |
| --- | --- |
| `0` | clear to ship (integrity valid, release-approved, generalized) |
| `1` | ship-blocked (integrity error, not release-approved, or overfit / unmeasured generalization) |
| `2` | environment/load failure (artifact missing/unreadable) or invalid arguments |

The `--format json` payload is schema-versioned (`gate_schema_version`
`1.0`), sorted-key, byte-stable for CI diffing. Python callers can use
`omegaprompt.run_gate(path) -> GateResult`.

> Note: a `gate` **MCP tool was intentionally not added**. The MCP tool set
> is frozen at 8 and the repository consistency contract enforces that count;
> `gate` is available via the CLI and the Python API only.

## `--format json` on `report` and `diff`

* `omegaprompt report --format json` emits a stable, schema-versioned
  summary projection (`summary_schema_version` `1.0`) — including the
  prominent overfit block — instead of dumping the full artifact.
* `omegaprompt diff --format json` exposes the existing structured
  `ArtifactDiff` as deterministic, sorted-key JSON. The regression verdict
  still drives the exit code (use `--no-fail-on-regression` to inspect
  without failing).

Both default to `markdown`, so existing invocations are unchanged.

## HTML report

`omegaprompt report --format html` renders a self-contained single-file
scorecard (inline CSS, **no JavaScript, no external assets — stdlib only**)
with status, the overfit verdict, summary metrics, sensitivity ranking, and
parameters. Suitable for attaching to a CI artifact bucket.

## Overfit metrics surfacing — "is my prompt overfit?"

The numbers that answer this already live inside the walk-forward block:
the **transfer correlation** (KC-4, per-item train↔holdout Pearson r) and
the **generalization gap**. 2.1.0 surfaces them as one prominent,
machine-readable block via `extract_overfit_metrics(artifact) ->
OverfitMetrics`, with a coarse `overfit_verdict`:

| verdict | meaning |
| --- | --- |
| `GENERALIZES` | the pre-declared walk-forward gate passed |
| `OVERFIT` | the gate failed (gap and/or transfer correlation below threshold) |
| `UNVERIFIABLE` | the gate did not pass but neither failure arm could be substantiated |
| `UNKNOWN` | no walk-forward block to read |

This is a **pure read**: it adds **no** field to `CalibrationArtifact`, so
the schema stays `2.0` and every golden hash is byte-stable. The block is
embedded in `report --format json` and the `gate` JSON output.

## `ollama` provider (distinct named adapter)

`make_provider("ollama")` now returns a clearly-named `OllamaProvider` with
Ollama defaults (local `http://localhost:11434/v1` base URL, keyless)
instead of a generic `local` adapter configured with a `backend` string.
This is backward-compatible: same name, same OpenAI-compatible transport,
same LOCAL/experimental capability reporting. Supported providers are now
`anthropic / openai / gemini / local / ollama / vllm / llama_cpp`.

## GitHub composite Action

`action.yml` at the repo root wraps `omegaprompt gate` so downstream repos
can gate prompts in CI:

```yaml
- uses: hibou04-ops/omegaprompt@v2.1.0
  with:
    artifact: path/to/artifact.json
    format: json
    require-generalization: "true"
```

Inputs: `artifact` (required), `format`, `require-generalization`,
`python-version`, `version`. Outputs: `passed`, `exit-code`. A complete
example workflow lives at [`examples/ci/ship-gate.yml`](../examples/ci/ship-gate.yml).

## Determinism / replay hardening

The offline golden-reference replay now also asserts the new gate JSON and
report-summary JSON are byte-stable across repeats and across a save→load
roundtrip, so CI catches any nondeterminism in the new surfaces without the
network.

## Tooling

* `publish.yml` is now **version-agnostic**: the verify step reads the
  single source of truth (`[project].version` via `tomllib`), asserts
  `__init__.__version__` matches, and (on a release) asserts the tag equals
  `v<version>`.
* The repository consistency checker's `EXPECTED.cli_commands` was extended
  with `gate`, and the badge-composition check was decoupled from the
  literal PyPI version (the version is owned by the separate badge-version
  check) so release bumps no longer false-trip composition.
