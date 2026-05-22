# Release Scope Freeze

This file records what must not expand during release-readiness work.

## Frozen Identity Boundaries

- GitHub repository: `hibou04-ops/omegaprompt`
- PyPI distribution: `omegaprompt`
- Primary import package: `omegaprompt`
- Compatibility import package and CLI alias: `omegacal`
- CLI executables: `omegaprompt`, `omegacal`, `omegaprompt-mcp`
- MCP optional extra: `mcp`, installed as `omegaprompt[mcp]`

`omegacal` is an alias for compatibility. It is not the PyPI distribution name.

## Frozen Trust Boundaries

Release work must not weaken:

- `CalibrationArtifact` schema validation
- provider capability checks
- train/test split discipline
- holdout and KC4 checks
- gate semantics
- diff regression behavior
- guarded vs expedition boundaries
- MCP optional-extra boundaries
- README/PyPI-safe Markdown link checks

`FAIL_KC4_GATE`, hard-gate failure, and CI-oriented `hold`/`block`
recommendations remain release blockers.

## Frozen Evidence Boundaries

Default tests and default CI must not call live providers. Optional live
provider tests must remain skipped unless explicitly enabled with:

```bash
OMEGAPROMPT_LIVE_PROVIDER_TESTS=1
```

Exact benchmark aggregates, provider superiority, model superiority, prompt
superiority, download counts, adoption counts, and exact test-count prose are
not allowed unless backed by the claim ledger and a valid source of truth,
generated document, reproducible command, deterministic artifact, or explicit
qualitative marker.

## Frozen Product Scope

Release readiness tooling is local and file-based. It may produce human and
JSON reports. It must not add:

- dashboard code
- hosted web application code
- proprietary release hosting
- tag mutation
- PyPI upload
- GitHub Release creation or editing

The readiness tools can report missing release state, including "tag exists but
GitHub Release marker missing", but they must not repair that state by
mutation.

Post-release verification follows the same freeze: `--local-only` verifies
local built artifacts without network, while `--network` may query PyPI/GitHub
after a human release but must not publish packages, push tags, or create/edit
GitHub Releases.
