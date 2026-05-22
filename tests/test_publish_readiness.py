from __future__ import annotations

import json
from pathlib import Path

from tools import publish_readiness, release_audit


def _fake_audit(status: str) -> dict[str, object]:
    blocking = status != "READY"
    check_status = "NOT_READY" if blocking else "OK"
    return {
        "schema_version": "1.0",
        "tool": "tools/release_audit.py",
        "root": "C:/repo",
        "final_status": status,
        "summary": {"total_checks": 1, "blocking_checks": 1 if blocking else 0, "status_counts": {check_status: 1}},
        "checks": [
            {
                "id": "FAKE",
                "status": check_status,
                "severity": "ERROR" if blocking else "INFO",
                "category": "test",
                "message": "fake",
                "details": {},
                "remediation": "fix" if blocking else None,
                "blocking": blocking,
            }
        ],
        "mutations": {
            "pypi_publish": False,
            "git_tags_created": False,
            "git_tags_pushed": False,
            "github_releases_created_or_edited": False,
        },
    }


def test_publish_readiness_writes_json_report(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(release_audit, "run_release_audit", lambda root, include_wheel=True: _fake_audit("READY"))
    output = tmp_path / "publish_readiness.json"

    exit_code = publish_readiness.main(["--strict", "--json-output", str(output)], root=tmp_path)

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["final_status"] == "READY"
    assert report["mutations"]["pypi_publish"] is False
    assert report["mutations"]["github_releases_created_or_edited"] is False


def test_publish_readiness_strict_fails_when_not_ready(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(release_audit, "run_release_audit", lambda root, include_wheel=True: _fake_audit("NOT_READY"))

    exit_code = publish_readiness.main(["--strict"], root=tmp_path)

    assert exit_code == 1


def test_publish_readiness_strict_classifies_tooling_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(release_audit, "run_release_audit", lambda root, include_wheel=True: _fake_audit("TOOLING_MISSING"))

    exit_code = publish_readiness.main(["--strict"], root=tmp_path)

    assert exit_code == 2
