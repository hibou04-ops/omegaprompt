from __future__ import annotations

import json
from pathlib import Path

from tools import generate_release_draft


ROOT = Path(__file__).resolve().parents[1]


def _sample_audit() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "tool": "tools/release_audit.py",
        "root": "C:/repo",
        "final_status": "TOOLING_MISSING",
        "version": {"pyproject_version": "1.7.4"},
        "summary": {"total_checks": 3, "blocking_checks": 1, "status_counts": {"OK": 2, "TOOLING_MISSING": 1}},
        "checks": [
            {"id": "VERSION_ALIGNMENT", "status": "OK", "message": "aligned", "blocking": False, "details": {}},
            {
                "id": "GIT_TAG_RELEASE_STATE",
                "status": "OK",
                "message": "Local tag exists; GitHub Release existence is deferred to post-release network verification.",
                "blocking": False,
                "details": {"tag": "v1.7.4", "local_tag_exists": True, "release_marker": None},
            },
            {"id": "WHEEL_BUILD", "status": "TOOLING_MISSING", "message": "build backend missing", "blocking": True, "details": {}},
        ],
        "deferred_external_checks": [
            {
                "id": "GITHUB_RELEASE_NETWORK_VERIFICATION",
                "status": "DEFERRED",
                "category": "github",
                "message": "GitHub Release existence is not verified by local release_audit.",
                "command": "python tools/post_release_verify.py --version 1.7.4 --network --json-output build/post_release_verify_network.json",
                "required_after_release": True,
                "network_required": True,
                "mutates_release_surfaces": False,
            }
        ],
        "mutations": {
            "pypi_publish": False,
            "git_tags_created": False,
            "git_tags_pushed": False,
            "github_releases_created_or_edited": False,
        },
    }


def _sample_readiness(audit: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "tool": "tools/publish_readiness.py",
        "root": "C:/repo",
        "final_status": "TOOLING_MISSING",
        "summary": audit["summary"],
        "release_audit": audit,
        "mutations": audit["mutations"],
    }


def test_render_release_draft_is_deterministic_and_claim_safe() -> None:
    audit = _sample_audit()
    ledger = json.loads((ROOT / "docs/claims/public_claim_ledger.json").read_text(encoding="utf-8"))
    reference = json.loads((ROOT / "examples/reference/reference_artifact.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "examples/reference/golden_manifest.json").read_text(encoding="utf-8"))

    first = generate_release_draft.render_release_draft(
        version="1.7.4",
        changelog_entry="### Fixed\n\n- Local release note.",
        ledger=ledger,
        audit=audit,
        readiness=_sample_readiness(audit),
        reference=reference,
        manifest=manifest,
    )
    second = generate_release_draft.render_release_draft(
        version="1.7.4",
        changelog_entry="### Fixed\n\n- Local release note.",
        ledger=ledger,
        audit=audit,
        readiness=_sample_readiness(audit),
        reference=reference,
        manifest=manifest,
    )

    assert first == second
    assert "PyPI publish performed by this generator: `false`" in first
    assert "GitHub Release created or edited by this generator: `false`" in first
    assert "GitHub Release marker: `missing`" in first
    assert "Deferred External Verification" in first
    assert "post_release_verify.py --version 1.7.4 --network" in first
    assert "PyPI state: `not queried; no publish performed`" in first
    assert "examples/reference/reference_artifact.json" in first
    assert "best provider" not in first.lower()
    assert "download count" not in first.lower()


def test_cli_generates_markdown_from_existing_json(tmp_path: Path) -> None:
    audit = _sample_audit()
    readiness = _sample_readiness(audit)
    audit_path = tmp_path / "release_audit.json"
    readiness_path = tmp_path / "publish_readiness.json"
    output = tmp_path / "release_draft.md"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")

    exit_code = generate_release_draft.main(
        [
            "--version",
            "1.7.4",
            "--release-audit-json",
            str(audit_path),
            "--publish-readiness-json",
            str(readiness_path),
            "--output",
            str(output),
            "--no-refresh-reports",
        ]
    )

    assert exit_code == 0
    text = output.read_text(encoding="utf-8")
    assert text.startswith("# omegaprompt 1.7.4 Release Draft")
    assert "WHEEL_BUILD" in text


def test_pr_template_requires_release_and_contract_checks() -> None:
    text = (ROOT / ".github/pull_request_template.md").read_text(encoding="utf-8")
    required = [
        "claim ledger",
        "New behavior includes tests",
        "Default tests and default CI do not add live provider/API calls",
        "artifact integrity checks",
        "provider contract tests",
        "MCP contract tests",
        "wheel build and core/MCP wheel smoke",
        "No PyPI publish, tag push, or GitHub Release create/edit action was performed",
    ]
    for phrase in required:
        assert phrase in text


def test_issue_templates_do_not_require_live_provider_defaults() -> None:
    template_dir = ROOT / ".github" / "ISSUE_TEMPLATE"
    for name in ("bug_report.yml", "prompt_ci_regression.yml", "provider_adapter.yml"):
        text = (template_dir / name).read_text(encoding="utf-8")
        assert "OMEGAPROMPT_LIVE_PROVIDER_TESTS=1" in text or "live-provider requirements" in text
        assert "required" in text
        assert "publish to PyPI" not in text or "not asking" in text
