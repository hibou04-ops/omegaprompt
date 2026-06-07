from __future__ import annotations

from pathlib import Path

import pytest

from tools import check_markdown_links


ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def codes(report: dict) -> set[str]:
    return {check["code"] for check in report["checks"]}


def test_valid_relative_file_link(tmp_path: Path) -> None:
    source = tmp_path / "docs" / "index.md"
    target = tmp_path / "docs" / "target.md"
    write(source, "[target](target.md)\n")
    write(target, "# Target\n")

    report = check_markdown_links.run_checks(tmp_path, scan_files=[source])

    assert report["summary"]["blocking_checks"] == 0
    assert report["summary"]["status_counts"] == {"OK": 1}


def test_missing_file_link_is_broken(tmp_path: Path) -> None:
    source = tmp_path / "docs" / "index.md"
    write(source, "[missing](missing.md)\n")

    report = check_markdown_links.run_checks(tmp_path, scan_files=[source])

    assert "MISSING_FILE" in codes(report)
    assert report["summary"]["blocking_checks"] == 1


def test_missing_anchor_is_broken(tmp_path: Path) -> None:
    source = tmp_path / "docs" / "index.md"
    target = tmp_path / "docs" / "target.md"
    write(source, "[missing anchor](target.md#missing-anchor)\n")
    write(target, "# Present Anchor\n")

    report = check_markdown_links.run_checks(tmp_path, scan_files=[source])

    assert "MISSING_ANCHOR" in codes(report)


def test_case_mismatch_path_is_broken(tmp_path: Path) -> None:
    source = tmp_path / "docs" / "index.md"
    target = tmp_path / "docs" / "Target.md"
    write(source, "[case mismatch](target.md)\n")
    write(target, "# Target\n")

    report = check_markdown_links.run_checks(tmp_path, scan_files=[source])

    assert "CASE_MISMATCH" in codes(report)


def test_windows_backslash_path_is_broken(tmp_path: Path) -> None:
    source = tmp_path / "docs" / "index.md"
    write(source, "[bad](subdir\\target.md)\n")

    report = check_markdown_links.run_checks(tmp_path, scan_files=[source])

    assert "WINDOWS_BACKSLASH" in codes(report)


def test_readme_pypi_unsafe_relative_cross_file_link_is_broken(tmp_path: Path) -> None:
    source = tmp_path / "README.md"
    write(source, "[한국어](README_KR.md)\n")
    write(tmp_path / "README_KR.md", "# Korean\n")

    report = check_markdown_links.run_checks(tmp_path, scan_files=[source])

    assert "PYPI_UNSAFE_RELATIVE_LINK" in codes(report)


def test_readme_pypi_safe_absolute_github_link_maps_to_local_file(tmp_path: Path) -> None:
    source = tmp_path / "README.md"
    write(source, "[한국어](https://github.com/hibou04-ops/omegaprompt/blob/main/README_KR.md)\n")
    write(tmp_path / "README_KR.md", "# Korean\n")

    report = check_markdown_links.run_checks(tmp_path, scan_files=[source])

    assert report["summary"]["blocking_checks"] == 0
    assert report["summary"]["status_counts"] == {"OK": 1}


def test_network_mode_is_disabled_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_urlopen(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("network should not be used by default")

    monkeypatch.setattr(check_markdown_links.urllib.request, "urlopen", fail_urlopen)
    source = tmp_path / "README.md"
    write(source, "[external](https://example.com/docs)\n")

    report = check_markdown_links.run_checks(tmp_path, scan_files=[source])

    assert report["network_enabled"] is False
    assert report["summary"]["blocking_checks"] == 0
    assert report["checks"][0]["details"]["network_checked"] is False


def test_readme_top_navigation_exposes_all_readme_variants() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    expected_links = [
        "https://github.com/hibou04-ops/omegaprompt/blob/main/README.md",
        "https://github.com/hibou04-ops/omegaprompt/blob/main/README_KR.md",
        "https://github.com/hibou04-ops/omegaprompt/blob/main/EASY_README.md",
        "https://github.com/hibou04-ops/omegaprompt/blob/main/EASY_README_KR.md",
    ]
    top = "\n".join(text.splitlines()[:25])
    for link in expected_links:
        assert link in top


def test_readme_top_badge_row_is_preserved_for_2_0_1() -> None:
    lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()

    assert lines[4:12] == [
        "[![CI](https://github.com/hibou04-ops/omegaprompt/actions/workflows/ci.yml/badge.svg)](https://github.com/hibou04-ops/omegaprompt/actions/workflows/ci.yml)",
        "[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)",
        "[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)",
        "[![PyPI](https://img.shields.io/badge/pypi-2.0.1-blue.svg)](https://pypi.org/project/omegaprompt/)",
        "[![Tests](https://img.shields.io/badge/tests-317%20passing-brightgreen.svg)](tests/)",
        "[![Artifact schema](https://img.shields.io/badge/artifact-schema%20v2.0-blueviolet.svg)](#8-the-calibrationartifact-schema-v20)",
        "[![MCP](https://img.shields.io/badge/MCP-server-blueviolet.svg)](#103-mcp-server-claude-code-cursor)",
        "[![Parent framework](https://img.shields.io/badge/framework-omega--lock-blueviolet.svg)](https://github.com/hibou04-ops/omega-lock)",
    ]


def test_readme_demo_video_and_replay_references_are_preserved() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "https://github.com/user-attachments/assets/d4308cc3-b8c1-4bb7-b67d-f763e6c26f11" in text
    assert "examples/demo_replay.py" in text
