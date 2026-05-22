from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"


def _ci_text() -> str:
    return CI.read_text(encoding="utf-8")


def test_ci_preserves_os_python_matrix_and_excludes_live_tests_by_default() -> None:
    text = _ci_text()

    assert "os: [ubuntu-latest, windows-latest]" in text
    assert 'python-version: ["3.11", "3.12"]' in text
    assert 'OMEGAPROMPT_LIVE_PROVIDER_TESTS: "0"' in text
    assert 'python -m pytest -q -m "not live"' in text
    assert "OMEGAPROMPT_LIVE_PROVIDER_TESTS=1" not in text


def test_ci_has_timeout_and_concurrency_controls() -> None:
    text = _ci_text()

    assert "concurrency:" in text
    assert "cancel-in-progress: true" in text
    for job in [
        "unit:",
        "generated-checks:",
        "reference-artifacts:",
        "provider-contract:",
        "mcp-contract:",
        "wheel-smoke:",
    ]:
        assert job in text
    assert text.count("timeout-minutes:") >= 6


def test_ci_runs_generated_reference_provider_mcp_and_wheel_checks() -> None:
    text = _ci_text()

    expected_commands = [
        "python tools/generate_readme_claims.py --check",
        "python tools/check_markdown_links.py --strict --json-output build/markdown_links.json",
        "python tools/check_repo_consistency.py --strict",
        "python tools/reproduce_golden_reference.py --check",
        "omegaprompt check-artifact examples/reference/reference_artifact.json --strict",
        "tests/test_provider_contracts.py tests/test_providers.py",
        "tests/test_mcp_server.py tests/test_mcp_execution.py tests/test_mcp_contracts.py tests/test_core_without_mcp.py",
        "python -m build",
        "python tools/wheel_smoke.py --wheel dist/*.whl --mode core",
        "python tools/wheel_smoke.py --wheel dist/*.whl --mode mcp",
    ]
    for command in expected_commands:
        assert command in text


def test_ci_does_not_publish_or_mutate_release_state() -> None:
    lowered = _ci_text().lower()

    forbidden = [
        "twine upload",
        "gh release",
        "git tag",
        "create-release",
        "pypi-token",
        "id-token: write",
    ]
    for token in forbidden:
        assert token not in lowered
