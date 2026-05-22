## Summary

Describe the change and the public surface it affects.

## Required Checks

- [ ] Public docs changed: claim ledger and generated README claims were updated or checked.
- [ ] New behavior includes tests.
- [ ] Default tests and default CI do not add live provider/API calls.
- [ ] Artifact/schema code changed: artifact integrity checks and schema tests were run.
- [ ] Provider docs/code changed: provider contract tests were run.
- [ ] MCP surface changed: MCP contract tests were run and the optional-extra boundary remains intact.
- [ ] Packaging, scripts, or extras changed: wheel build and core/MCP wheel smoke were run or a TOOLING_MISSING/ENVIRONMENT_BLOCKED blocker is documented.
- [ ] README.md top badge composition is unchanged.
- [ ] No provider/model/prompt superiority, exact benchmark aggregate, exact README prose test-count, download, or adoption claim was added without ledger-backed evidence.
- [ ] No PyPI publish, tag push, or GitHub Release create/edit action was performed.

## Verification

```bash
python tools/check_repo_consistency.py --strict
python tools/generate_readme_claims.py --check
python -m pytest -q -m "not live"
```

Add any focused commands that apply to this PR.

## Release/Publish State

State whether `tools/release_audit.py` or `tools/publish_readiness.py` was run, and paste the final status if relevant.
