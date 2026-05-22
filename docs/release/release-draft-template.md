# Release Draft Template

Release drafts are generated locally with:

```bash
python tools/generate_release_draft.py --version <version> --output build/release_draft.md
```

The generator reads:

- `CHANGELOG.md`
- `docs/claims/public_claim_ledger.json`
- release audit JSON data
- publish readiness JSON data

By default the audit/readiness data is refreshed in memory and round-tripped
through JSON before rendering, so the only persistent output is the Markdown
draft. To render from preexisting report files instead, pass
`--no-refresh-reports` with `--release-audit-json` and
`--publish-readiness-json`.

## Generated Sections

The generated Markdown contains:

- release identity and readiness status
- release-state visibility, including local tag state and GitHub Release marker
  state as separate fields
- changelog excerpt for the requested version
- deterministic artifact references and reproducible commands
- verification commands
- known limitations from warning/blocking release gates
- explicit no-mutation statement for PyPI, tags, and GitHub Releases
- claim-ledger evidence references

## Scope Boundaries

The generator does not publish to PyPI, create or push tags, or create/edit a
GitHub Release. It produces Markdown only.

The draft must not add unsupported provider/model/prompt superiority claims,
exact benchmark aggregates, exact README prose test-count claims, download
counts, or adoption counts. Such claims require ledger-backed evidence and a
deterministic source.
