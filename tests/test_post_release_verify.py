from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path

from tools import post_release_verify


ROOT = Path(__file__).resolve().parents[1]


def _write_fixture_dist(root: Path, version: str = "1.7.4") -> None:
    dist = root / "dist"
    dist.mkdir()
    wheel = dist / f"omegaprompt-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr("omegaprompt/__init__.py", "")
        zf.writestr(
            f"omegaprompt-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.3\nName: omegaprompt\nVersion: {version}\n",
        )

    sdist = dist / f"omegaprompt-{version}.tar.gz"
    pyproject = f"[project]\nname = \"omegaprompt\"\nversion = \"{version}\"\n".encode("utf-8")
    with tarfile.open(sdist, "w:gz") as tf:
        info = tarfile.TarInfo(f"omegaprompt-{version}/pyproject.toml")
        info.size = len(pyproject)
        tf.addfile(info, io.BytesIO(pyproject))


def _patch_local_only_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(
        post_release_verify,
        "_check_generated_claims",
        lambda root: post_release_verify.ok("GENERATED_CLAIMS_FRESH", "fresh", category="claims"),
    )
    monkeypatch.setattr(
        post_release_verify,
        "_check_repository_consistency",
        lambda root: post_release_verify.ok("REPOSITORY_CONSISTENCY", "consistent", category="repo"),
    )
    monkeypatch.setattr(
        post_release_verify,
        "_check_local_wheel_smoke",
        lambda root, version: post_release_verify.ok("LOCAL_WHEEL_SMOKE", "smoke ok", category="local"),
    )
    monkeypatch.setattr(
        post_release_verify,
        "_check_release_audit_local_compat",
        lambda root: post_release_verify.ok("RELEASE_AUDIT_LOCAL_COMPAT", "audit ok", category="release"),
    )
    monkeypatch.setattr(
        post_release_verify,
        "_check_publish_readiness_local_compat",
        lambda root: post_release_verify.ok("PUBLISH_READINESS_LOCAL_COMPAT", "readiness ok", category="release"),
    )


def test_dry_run_writes_json_without_network(monkeypatch, tmp_path: Path) -> None:
    def blocked_fetch(url: str, *, timeout: int = 20):
        raise AssertionError("dry-run must not fetch network resources")

    monkeypatch.setattr(post_release_verify, "fetch_json", blocked_fetch)
    output = tmp_path / "post_release_verify.json"

    exit_code = post_release_verify.main(
        ["--version", "1.7.4", "--dry-run", "--json-output", str(output)],
        root=ROOT,
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["final_status"] == "READY"
    assert report["mode"] == {
        "dry_run": True,
        "local_only": False,
        "network": False,
        "network_checks_included": False,
    }
    assert report["mutations"]["pypi_publish"] is False
    assert report["mutations"]["github_releases_created_or_edited"] is False
    assert "SKIPPED" not in report["summary"]["status_counts"]
    assert not any(check["id"] == "NETWORK_CHECKS" for check in report["checks"])


def test_local_only_fixture_dist_returns_only_ok(monkeypatch, tmp_path: Path) -> None:
    _write_fixture_dist(tmp_path)
    _patch_local_only_dependencies(monkeypatch)
    monkeypatch.setattr(
        post_release_verify,
        "fetch_json",
        lambda url, timeout=20: (_ for _ in ()).throw(AssertionError("local-only must not fetch network resources")),
    )

    report = post_release_verify.run_verification(
        root=tmp_path,
        version="1.7.4",
        local_only=True,
    )

    assert report["final_status"] == "READY"
    assert report["summary"]["status_counts"] == {"OK": 7}
    assert {check["status"] for check in report["checks"]} == {"OK"}
    assert "WARNING" not in json.dumps(report)
    assert "SKIPPED" not in json.dumps(report)
    assert report["mode"]["network_checks_included"] is False
    assert "NETWORK_CHECKS" not in {check["id"] for check in report["checks"]}


def test_local_only_missing_dist_is_not_ready_not_warning(monkeypatch, tmp_path: Path) -> None:
    _patch_local_only_dependencies(monkeypatch)

    report = post_release_verify.run_verification(
        root=tmp_path,
        version="1.7.4",
        local_only=True,
    )

    by_id = {check["id"]: check for check in report["checks"]}
    assert report["final_status"] == "NOT_READY"
    assert by_id["LOCAL_DIST_ARTIFACTS"]["status"] == "NOT_VERIFIED"
    assert "WARNING" not in report["summary"]["status_counts"]


def test_local_only_cli_does_not_count_disabled_network_as_skipped(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_fixture_dist(tmp_path)
    _patch_local_only_dependencies(monkeypatch)
    output = tmp_path / "post_release_verify.json"

    exit_code = post_release_verify.main(
        ["--version", "1.7.4", "--local-only", "--json-output", str(output)],
        root=tmp_path,
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["status_counts"] == {"OK": 7}
    assert "SKIPPED" not in report["summary"]["status_counts"]
    assert not any(check["id"] == "NETWORK_CHECKS" for check in report["checks"])


def test_final_status_prioritizes_blocked_and_tooling_states() -> None:
    assert post_release_verify.final_status(
        [post_release_verify.ok("OK", "ok", category="x")],
        network=False,
    ) == "READY"
    assert post_release_verify.final_status(
        [post_release_verify.ok("OK", "ok", category="x")],
        network=True,
    ) == "VERIFIED"
    assert post_release_verify.final_status(
        [post_release_verify.not_verified("N", "no", category="x")],
        network=True,
    ) == "NOT_VERIFIED"
    assert post_release_verify.final_status(
        [post_release_verify.not_verified("N", "no", category="x")],
        network=False,
        local_only=True,
    ) == "NOT_READY"
    assert post_release_verify.final_status(
        [post_release_verify.tooling_missing("T", "missing", category="x")],
        network=True,
    ) == "TOOLING_MISSING"
    assert post_release_verify.final_status(
        [
            post_release_verify.tooling_missing("T", "missing", category="x"),
            post_release_verify.environment_blocked("E", "blocked", category="x"),
        ],
        network=True,
    ) == "ENVIRONMENT_BLOCKED"


def test_network_success_with_mocked_pypi_and_github(monkeypatch) -> None:
    def fake_fetch(url: str, *, timeout: int = 20):
        if "pypi.org" in url:
            return {
                "info": {"name": "omegaprompt", "version": "1.7.4", "description": "Provider-neutral prompt calibration."},
                "urls": [
                    {"filename": "omegaprompt-1.7.4-py3-none-any.whl", "packagetype": "bdist_wheel"},
                    {"filename": "omegaprompt-1.7.4.tar.gz", "packagetype": "sdist"},
                ],
            }
        if "/git/ref/tags/" in url:
            return {"ref": "refs/tags/v1.7.4"}
        if "/releases/tags/" in url:
            return {"tag_name": "v1.7.4", "html_url": "https://github.example/release"}
        raise AssertionError(url)

    monkeypatch.setattr(post_release_verify, "fetch_json", fake_fetch)
    monkeypatch.setattr(
        post_release_verify,
        "_check_pypi_install",
        lambda version: [
            post_release_verify.ok("PYPI_CORE_INSTALL", "core ok", category="install"),
            post_release_verify.ok("PYPI_MCP_EXTRA_INSTALL", "mcp ok", category="install"),
        ],
    )

    report = post_release_verify.run_verification(root=ROOT, version="1.7.4", network=True)

    assert report["final_status"] == "VERIFIED"
    by_id = {check["id"]: check for check in report["checks"]}
    assert by_id["PYPI_PROJECT_VERSION"]["status"] == "OK"
    assert by_id["PYPI_DISTRIBUTION_METADATA"]["status"] == "OK"
    assert by_id["GITHUB_TAG"]["status"] == "OK"
    assert by_id["GITHUB_RELEASE"]["status"] == "OK"


def test_network_success_is_not_blocked_by_local_dist_mismatch(monkeypatch, tmp_path: Path) -> None:
    _patch_local_only_dependencies(monkeypatch)
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "omegacal-1.7.4-py3-none-any.whl").write_text("", encoding="utf-8")

    def fake_fetch(url: str, *, timeout: int = 20):
        if "pypi.org" in url:
            return {
                "info": {"name": "omegaprompt", "version": "1.7.4", "description": "Provider-neutral prompt calibration."},
                "urls": [
                    {"filename": "omegaprompt-1.7.4-py3-none-any.whl", "packagetype": "bdist_wheel"},
                    {"filename": "omegaprompt-1.7.4.tar.gz", "packagetype": "sdist"},
                ],
            }
        if "/git/ref/tags/" in url:
            return {"ref": "refs/tags/v1.7.4"}
        if "/releases/tags/" in url:
            return {"tag_name": "v1.7.4", "html_url": "https://github.example/release"}
        raise AssertionError(url)

    monkeypatch.setattr(post_release_verify, "fetch_json", fake_fetch)
    monkeypatch.setattr(
        post_release_verify,
        "_check_pypi_install",
        lambda version: [
            post_release_verify.ok("PYPI_CORE_INSTALL", "core ok", category="install"),
            post_release_verify.ok("PYPI_MCP_EXTRA_INSTALL", "mcp ok", category="install"),
        ],
    )

    report = post_release_verify.run_verification(root=tmp_path, version="1.7.4", network=True)

    assert report["final_status"] == "VERIFIED"
    assert "LOCAL_DIST_ARTIFACTS" not in {check["id"] for check in report["checks"]}


def test_pypi_version_mismatch_is_not_verified(monkeypatch) -> None:
    monkeypatch.setattr(
        post_release_verify,
        "fetch_json",
        lambda url, timeout=20: {
            "info": {"name": "omegaprompt", "version": "1.7.3", "description": ""},
            "urls": [],
        },
    )
    monkeypatch.setattr(post_release_verify, "_check_pypi_install", lambda version: [])

    report = post_release_verify.run_verification(root=ROOT, version="1.7.4", network=True)

    assert report["final_status"] == "NOT_VERIFIED"
    assert any(check["id"] == "PYPI_PROJECT_VERSION" and check["status"] == "NOT_VERIFIED" for check in report["checks"])


def test_network_block_is_environment_blocked(monkeypatch) -> None:
    def fake_fetch(url: str, *, timeout: int = 20):
        raise post_release_verify.NetworkBlocked("DNS blocked")

    monkeypatch.setattr(post_release_verify, "fetch_json", fake_fetch)
    monkeypatch.setattr(post_release_verify, "_check_pypi_install", lambda version: [])

    report = post_release_verify.run_verification(root=ROOT, version="1.7.4", network=True)

    assert report["final_status"] == "ENVIRONMENT_BLOCKED"
    assert any(check["status"] == "ENVIRONMENT_BLOCKED" for check in report["checks"])


def test_github_release_missing_is_separate_from_tag(monkeypatch) -> None:
    def fake_fetch(url: str, *, timeout: int = 20):
        if "pypi.org" in url:
            return {
                "info": {"name": "omegaprompt", "version": "1.7.4", "description": ""},
                "urls": [
                    {"filename": "omegaprompt-1.7.4-py3-none-any.whl"},
                    {"filename": "omegaprompt-1.7.4.tar.gz"},
                ],
            }
        if "/git/ref/tags/" in url:
            return {"ref": "refs/tags/v1.7.4"}
        if "/releases/tags/" in url:
            raise post_release_verify.ResourceMissing("release missing")
        raise AssertionError(url)

    monkeypatch.setattr(post_release_verify, "fetch_json", fake_fetch)
    monkeypatch.setattr(
        post_release_verify,
        "_check_pypi_install",
        lambda version: [
            post_release_verify.ok("PYPI_CORE_INSTALL", "core ok", category="install"),
            post_release_verify.ok("PYPI_MCP_EXTRA_INSTALL", "mcp ok", category="install"),
        ],
    )

    report = post_release_verify.run_verification(root=ROOT, version="1.7.4", network=True)
    by_id = {check["id"]: check for check in report["checks"]}

    assert report["final_status"] == "NOT_VERIFIED"
    assert by_id["GITHUB_TAG"]["status"] == "OK"
    assert by_id["GITHUB_RELEASE"]["status"] == "NOT_VERIFIED"
    assert by_id["GITHUB_RELEASE"]["details"]["release_missing"] is True


def test_local_dist_mismatch_blocks_when_artifacts_exist(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "omegaprompt-1.7.3-py3-none-any.whl").write_text("", encoding="utf-8")

    check = post_release_verify._check_local_dist_artifacts(tmp_path, "1.7.4", required=True)

    assert check.status == "NOT_VERIFIED"
    assert check.blocking is True


def test_package_metadata_parser_does_not_overwrite_header_identity() -> None:
    metadata = post_release_verify._parse_package_metadata(
        "Metadata-Version: 2.4\n"
        "Name: omegaprompt\n"
        "Version: 1.7.4\n"
        "Description-Content-Type: text/markdown\n"
        "\n"
        "Example text with a later Name: str line from a license or README.\n"
    )

    assert metadata == {"name": "omegaprompt", "version": "1.7.4"}


def test_release_docs_reference_post_release_verifier() -> None:
    text = (ROOT / "docs/release/release-checklist.md").read_text(encoding="utf-8")

    assert "tools/post_release_verify.py" in text
    assert "--network" in text
    assert "--dry-run" in text
    assert "--local-only" in text
    assert 'python -m pytest -q -m "not live"' in text
    assert "does not publish to PyPI" in text
    assert "informational" in text


def test_canonical_no_network_pytest_excludes_live_marker() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    release_checklist = (ROOT / "docs/release/release-checklist.md").read_text(encoding="utf-8")
    release_draft_tool = (ROOT / "tools/generate_release_draft.py").read_text(encoding="utf-8")
    pr_template = (ROOT / ".github/pull_request_template.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert '"live: opt-in provider smoke tests' in pyproject
    for text in (release_checklist, release_draft_tool, pr_template, readme):
        assert 'python -m pytest -q -m "not live"' in text
