# Release Checklist

This checklist is for local release readiness only. It does not publish to
PyPI, create tags, push tags, or create/edit GitHub Releases.

## Required Local Gates

Run these from the repository root:

```bash
python tools/release_audit.py --strict
python tools/publish_readiness.py --strict --json-output build/publish_readiness.json
python tools/check_markdown_links.py --strict --json-output build/markdown_links.json
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
- README/PyPI-safe Markdown link integrity, including local file targets,
  anchors, case-sensitive paths, and no-network default behavior
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
present, the report keeps local tag visibility as an `OK` local check and
lists GitHub Release existence under deferred external checks. Verify that
external state after the actual release with:

```bash
python tools/post_release_verify.py --version <version> --network --json-output build/post_release_verify_network.json
```

## Wheel Boundary

The audit builds a wheel into a temporary local directory and then runs:

```bash
python tools/wheel_smoke.py --wheel <built-wheel> --mode core
python tools/wheel_smoke.py --wheel <built-wheel> --mode mcp
```

The smoke checks install from the local wheel only. They do not rely on a
globally installed `omegaprompt`, and they do not call live providers.

## Final Local Verification

The final no-network verification gate requires local release artifacts. Build
the wheel and sdist first, then run local-only post-release verification:

```bash
python -m build
python tools/post_release_verify.py --version <version> --local-only --json-output build/post_release_verify.json
python -m pytest -q -m "not live"
```

`--local-only` does not contact PyPI or GitHub and does not add disabled
network checks as skipped report rows. It requires
`dist/omegaprompt-<version>-py3-none-any.whl` and
`dist/omegaprompt-<version>.tar.gz`, inspects their names and metadata, and
delegates local wheel smoke coverage for core and MCP boundaries. The happy
path should contain only `OK` checks.

Focused wheel smoke remains useful for debugging packaging failures:

```bash
python tools/wheel_smoke.py --wheel dist/*.whl --mode core
python tools/wheel_smoke.py --wheel dist/*.whl --mode mcp
```

## Post-Release Network Verification

After a human completes an out-of-band release, verify the published surfaces
without mutating them:

```bash
python tools/post_release_verify.py --version <version> --network --json-output build/post_release_verify_network.json
```

`--network` checks PyPI/GitHub and isolated PyPI installs. Network, DNS, auth,
or rate-limit failures are `ENVIRONMENT_BLOCKED`, not verification success.
The verifier does not publish to PyPI, create or push tags, or create/edit
GitHub Releases.

## Informational Dry Run

For an informational dry run that does not contact PyPI or GitHub:

```bash
python tools/post_release_verify.py --version <version> --dry-run --json-output build/post_release_verify.json
```

`--dry-run` is not the complete final local gate. It may report `WARNING` for
missing non-blocking checks, but it does not require local dist artifacts and
does not contact PyPI or GitHub. Use `--local-only` after `python -m build` for
a clean local release gate.

The post-release verifier reports `READY` for a clean local-only gate and
`VERIFIED` only when network checks are enabled and all required
PyPI/GitHub/install checks pass. `TOOLING_MISSING` and `ENVIRONMENT_BLOCKED`
remain blockers, not verification success.
