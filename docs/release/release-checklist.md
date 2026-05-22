# Release Checklist

This checklist is for local release readiness only. It does not publish to
PyPI, create tags, push tags, or create/edit GitHub Releases.

## Required Local Gates

Run these from the repository root:

```bash
python tools/release_audit.py --strict
python tools/publish_readiness.py --strict --json-output build/publish_readiness.json
```

The readiness status must be `READY` before a human starts any separate
publish procedure.

The audit checks:

- version alignment across `pyproject.toml`, `src/omegaprompt/__init__.py`,
  the README PyPI badge, and `CHANGELOG.md`
- branch cleanliness visibility
- generated README claim freshness
- claim ledger validation and unsupported public-claim scans
- deterministic golden reference artifact freshness
- reference artifact integrity
- provider capability docs against capability code
- README.md top badge composition
- default CI/live-provider boundaries
- repository consistency checker output
- local tag and local release-marker visibility
- local wheel build
- core and MCP wheel smoke checks

## Status Meanings

- `READY`: all blocking gates passed.
- `NOT_READY`: a repository, artifact, claim, docs, build, or smoke gate failed.
- `TOOLING_MISSING`: local tooling required to evaluate readiness is missing.
- `ENVIRONMENT_BLOCKED`: local filesystem or environment access prevented a
  readiness decision.

`TOOLING_MISSING` and `ENVIRONMENT_BLOCKED` are never release approval.

## Release State Reporting

The audit reports local tag state for `v<pyproject version>` and looks for a
local release marker under:

- `.github/releases/v<version>.md`
- `docs/release/releases/v<version>.md`
- `docs/release/v<version>.md`

This is visibility only. The audit does not call GitHub, create tags, push
tags, or create GitHub Releases. If a tag exists but no release marker is
present, the report makes that visible for a human to resolve out of band.

## Wheel Boundary

The audit builds a wheel into a temporary local directory and then runs:

```bash
python tools/wheel_smoke.py --wheel <built-wheel> --mode core
python tools/wheel_smoke.py --wheel <built-wheel> --mode mcp
```

The smoke checks install from the local wheel only. They do not rely on a
globally installed `omegaprompt`, and they do not call live providers.

