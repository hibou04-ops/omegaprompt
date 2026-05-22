from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tools import release_audit


ROOT = Path(__file__).resolve().parents[1]


def test_final_status_prioritizes_environment_and_tooling_failures() -> None:
    assert release_audit.final_status([release_audit.ok("OK", "ok", category="x")]) == "READY"
    assert release_audit.final_status([release_audit.not_ready("N", "no", category="x")]) == "NOT_READY"
    assert release_audit.final_status([release_audit.tooling_missing("T", "missing", category="x")]) == "TOOLING_MISSING"
    assert release_audit.final_status(
        [
            release_audit.tooling_missing("T", "missing", category="x"),
            release_audit.environment_blocked("E", "blocked", category="x"),
        ]
    ) == "ENVIRONMENT_BLOCKED"


def test_stale_generated_claims_block_release(monkeypatch) -> None:
    monkeypatch.setattr(
        release_audit,
        "_check_generated_claims",
        lambda root: release_audit.not_ready("README_CLAIMS_FRESH", "stale", category="claims"),
    )

    report = release_audit.run_release_audit(ROOT, include_wheel=False)

    assert report["final_status"] == "NOT_READY"
    assert any(check["id"] == "README_CLAIMS_FRESH" and check["blocking"] for check in report["checks"])


def test_invalid_claim_ledger_blocks_release(monkeypatch) -> None:
    monkeypatch.setattr(
        release_audit,
        "_check_claim_ledger",
        lambda root: release_audit.not_ready("CLAIM_LEDGER_VALID", "unsupported claim", category="claims"),
    )

    report = release_audit.run_release_audit(ROOT, include_wheel=False)

    assert report["final_status"] == "NOT_READY"
    assert any(check["id"] == "CLAIM_LEDGER_VALID" and check["blocking"] for check in report["checks"])


def test_stale_reference_artifacts_block_release(monkeypatch) -> None:
    monkeypatch.setattr(
        release_audit,
        "_check_reference_artifacts",
        lambda root: release_audit.not_ready("REFERENCE_ARTIFACTS_FRESH", "stale", category="artifact"),
    )

    report = release_audit.run_release_audit(ROOT, include_wheel=False)

    assert report["final_status"] == "NOT_READY"
    assert any(check["id"] == "REFERENCE_ARTIFACTS_FRESH" and check["blocking"] for check in report["checks"])


def test_wheel_smoke_failure_blocks_release(monkeypatch, tmp_path: Path) -> None:
    wheel = tmp_path / "omegaprompt-1.7.4-py3-none-any.whl"
    wheel.write_text("fake", encoding="utf-8")
    monkeypatch.setattr(
        release_audit,
        "_check_wheel_build",
        lambda root, outdir: (release_audit.ok("WHEEL_BUILD", "built", category="wheel"), wheel),
    )
    monkeypatch.setattr(
        release_audit,
        "_check_wheel_smoke",
        lambda wheel, mode: release_audit.not_ready(f"WHEEL_SMOKE_{mode.upper()}", "failed", category="wheel"),
    )

    report = release_audit.run_release_audit(ROOT, include_wheel=True)

    assert report["final_status"] == "NOT_READY"
    assert any(check["id"] == "WHEEL_SMOKE_CORE" and check["blocking"] for check in report["checks"])


def test_missing_build_tooling_is_not_release_approval(monkeypatch) -> None:
    monkeypatch.setattr(
        release_audit,
        "_check_wheel_build",
        lambda root, outdir: (
            release_audit.tooling_missing("WHEEL_BUILD", "build missing", category="wheel"),
            None,
        ),
    )

    report = release_audit.run_release_audit(ROOT, include_wheel=True)

    assert report["final_status"] == "TOOLING_MISSING"
    assert release_audit.strict_exit_code(report["final_status"]) == 2


def test_build_backend_failure_is_classified_as_tooling_missing() -> None:
    assert release_audit._looks_like_build_tooling_failure(
        "Backend 'hatchling.build' is not available."
    )
    assert release_audit._looks_like_build_tooling_failure(
        "Installing packages in isolated environment: hatchling"
    )


def test_clean_local_release_audit_has_only_ok_checks_and_deferred_external_visibility(
    monkeypatch,
    tmp_path: Path,
) -> None:
    version_facts = {
        "pyproject_version": "1.7.4",
        "package_version": "1.7.4",
        "readme_badge_version": "1.7.4",
        "changelog_latest_version": "1.7.4",
    }
    monkeypatch.setattr(release_audit, "_version_facts", lambda root: version_facts)
    monkeypatch.setattr(
        release_audit,
        "_check_branch_cleanliness",
        lambda root: release_audit.ok("GIT_BRANCH_STATE", "clean", category="git"),
    )
    for name, check_id, category in (
        ("_check_claim_ledger", "CLAIM_LEDGER_VALID", "claims"),
        ("_check_generated_claims", "README_CLAIMS_FRESH", "claims"),
        ("_check_reference_artifacts", "REFERENCE_ARTIFACTS_FRESH", "artifact"),
        ("_check_artifact_integrity", "REFERENCE_ARTIFACT_INTEGRITY", "artifact"),
        ("_check_provider_docs_code", "PROVIDER_DOCS_CODE_CONSISTENCY", "providers"),
        ("_check_readme_badges", "README_BADGE_COMPOSITION", "docs"),
        ("_check_markdown_links", "MARKDOWN_LINKS", "docs"),
        ("_check_no_default_live_tests", "DEFAULT_CI_NO_LIVE_TESTS", "ci"),
        ("_check_repository_consistency", "REPOSITORY_CONSISTENCY", "repo"),
    ):
        monkeypatch.setattr(
            release_audit,
            name,
            lambda root, check_id=check_id, category=category: release_audit.ok(check_id, "ok", category=category),
        )

    def fake_git(root: Path, args: list[str]):
        assert args == ["tag", "--list", "v1.7.4"]
        return SimpleNamespace(returncode=0, stdout="v1.7.4\n", stderr="")

    wheel = tmp_path / "omegaprompt-1.7.4-py3-none-any.whl"
    wheel.write_text("fake", encoding="utf-8")
    monkeypatch.setattr(release_audit, "_git", fake_git)
    monkeypatch.setattr(
        release_audit,
        "_check_wheel_build",
        lambda root, outdir: (release_audit.ok("WHEEL_BUILD", "built", category="wheel"), wheel),
    )
    monkeypatch.setattr(
        release_audit,
        "_check_wheel_smoke",
        lambda wheel, mode: release_audit.ok(f"WHEEL_SMOKE_{mode.upper()}", "smoke ok", category="wheel"),
    )

    report = release_audit.run_release_audit(tmp_path, include_wheel=True)
    human = release_audit.render_human(report)

    assert report["final_status"] == "READY"
    assert report["summary"]["blocking_checks"] == 0
    assert report["summary"]["status_counts"] == {"OK": 15}
    assert {check["status"] for check in report["checks"]} == {"OK"}
    assert "WARNING" not in report["summary"]["status_counts"]
    assert "SKIPPED" not in report["summary"]["status_counts"]
    assert report["mutations"]["github_releases_created_or_edited"] is False
    assert report["mutations"]["git_tags_pushed"] is False
    deferred = report["deferred_external_checks"]
    assert deferred[0]["id"] == "GITHUB_RELEASE_NETWORK_VERIFICATION"
    assert deferred[0]["status"] == "DEFERRED"
    assert "python tools/post_release_verify.py --version 1.7.4 --network --json-output build/post_release_verify_network.json" in deferred[0]["command"]
    assert "Deferred external checks:" in human
    assert deferred[0]["command"] in human
    assert "GITHUB_RELEASE_NETWORK_VERIFICATION" not in {check["id"] for check in report["checks"]}


def test_dirty_working_tree_still_reports_branch_warning(monkeypatch, tmp_path: Path) -> None:
    def fake_git(root: Path, args: list[str]):
        assert args == ["status", "--porcelain", "--branch"]
        return SimpleNamespace(returncode=0, stdout="## main\n M README.md\n", stderr="")

    monkeypatch.setattr(release_audit, "_git", fake_git)

    check = release_audit._check_branch_cleanliness(tmp_path)

    assert check.status == "WARNING"
    assert check.id == "GIT_BRANCH_STATE"


def test_tag_exists_without_release_marker_is_visible(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path
    (root / "docs" / "release").mkdir(parents=True)

    def fake_git(root: Path, args: list[str]):
        assert args == ["tag", "--list", "v1.7.4"]
        return SimpleNamespace(returncode=0, stdout="v1.7.4\n", stderr="")

    monkeypatch.setattr(release_audit, "_git", fake_git)

    check = release_audit._check_git_tag_release_state(root, "1.7.4")

    assert check.status == "OK"
    assert check.details["local_tag_exists"] is True
    assert check.details["release_marker"] is None
    assert check.details["network_or_github_mutation"] is False
    assert "deferred" in check.message
