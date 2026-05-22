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


def test_tag_exists_without_release_marker_is_visible(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path
    (root / "docs" / "release").mkdir(parents=True)

    def fake_git(root: Path, args: list[str]):
        assert args == ["tag", "--list", "v1.7.4"]
        return SimpleNamespace(returncode=0, stdout="v1.7.4\n", stderr="")

    monkeypatch.setattr(release_audit, "_git", fake_git)

    check = release_audit._check_git_tag_release_state(root, "1.7.4")

    assert check.status == "WARNING"
    assert check.details["local_tag_exists"] is True
    assert check.details["release_marker"] is None
    assert check.details["network_or_github_mutation"] is False
