from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "check_repo_consistency.py"
SPEC = importlib.util.spec_from_file_location("check_repo_consistency", SCRIPT_PATH)
assert SPEC and SPEC.loader
checker = importlib.util.module_from_spec(SPEC)
sys.modules["check_repo_consistency"] = checker
SPEC.loader.exec_module(checker)


def write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(root: Path, rel: str, data: dict) -> None:
    write(root, rel, json.dumps(data, indent=2))


def make_consistent_repo(root: Path) -> None:
    write(
        root,
        "pyproject.toml",
        """
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "omegaprompt"
version = "2.0.2"
dependencies = [
  "typer>=0.12.0",
  "pydantic>=2.6.0",
  "omega-lock>=0.1.4",
  "anthropic>=0.40.0",
  "openai>=1.50.0",
  "google-genai>=1.0.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0"]
anthropic = ["anthropic>=0.40.0"]
openai = ["openai>=1.50.0"]
gemini = ["google-genai>=1.0.0"]
mcp = ["mcp>=1.0.0"]

[project.urls]
Repository = "https://github.com/hibou04-ops/omegaprompt"

[project.scripts]
omegacal = "omegacal.cli:app"
omegaprompt = "omegaprompt.cli:app"
omegaprompt-mcp = "omegaprompt.mcp.__main__:main"

[tool.hatch.build.targets.wheel]
packages = ["src/omegaprompt", "src/omegacal"]
""".strip(),
    )
    write(
        root,
        "src/omegaprompt/__init__.py",
        '''
__version__ = "2.0.2"
__all__ = [
    "calibrate",
    "evaluate",
    "report",
    "diff",
    "measure_sensitivity",
    "grade",
    "preflight",
    "classify_traps",
]
'''.strip(),
    )
    write(root, "src/omegacal/__init__.py", "from omegaprompt import *  # noqa: F401,F403\n")
    write(root, "src/omegacal/cli.py", "from omegaprompt.cli import app\n")
    write(root, "src/omegacal/__main__.py", "from omegaprompt.cli import app\n\nif __name__ == '__main__':\n    app()\n")

    badges = "\n".join(
        [
            "[![CI](https://github.com/hibou04-ops/omegaprompt/actions/workflows/ci.yml/badge.svg)](https://github.com/hibou04-ops/omegaprompt/actions/workflows/ci.yml)",
            "[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)",
            "[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org)",
            "[![PyPI](https://img.shields.io/badge/pypi-2.0.2-blue.svg)](https://pypi.org/project/omegaprompt/)",
            "[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)",
            "[![Artifact schema](https://img.shields.io/badge/artifact-schema%20v2.0-blueviolet.svg)](#8-the-calibrationartifact-schema-v20)",
            "[![MCP](https://img.shields.io/badge/MCP-server-blueviolet.svg)](#103-mcp-server-claude-code-cursor)",
            "[![Parent framework](https://img.shields.io/badge/framework-omega--lock-blueviolet.svg)](https://github.com/hibou04-ops/omega-lock)",
        ]
    )
    write(
        root,
        "README.md",
        f"""# omegaprompt

{badges}

```bash
pip install omegaprompt
```

> **v2.0.2 (2026-06-08)** - current package version.

Exit codes: `0` when the artifact status is OK, `1` when the artifact status is not OK, `2` on argument or environment problems.

The `omegacal` CLI binary remains as a compatibility alias during migration.

The test suite runs with `pytest -q` and the badge records the current exact count.
""",
    )
    write(root, "README_KR.md", "# omegaprompt\n\n> **v2.0.2 (2026-06-08)** - current package version.\n")
    write(root, "EASY_README.md", "# omegaprompt Easy Start\n")
    write(root, "EASY_README_KR.md", "# omegaprompt Korean Easy Start\n")
    write(root, "CHANGELOG.md", "## [Unreleased]\n\n## [2.0.2] - 2026-06-08\n")
    write(
        root,
        "docs/provider-capabilities.md",
        """# Provider Capabilities

| Provider | Tier | Status | Judge suitability | Notes |
| --- | --- | --- | --- | --- |
| Gemini | Tier 2 cloud-grade | Implemented target adapter | Not ship-grade | Adapter implemented; judge remains unvalidated |

## Gemini

Current state:
- implemented target adapter
- placeholder flag is false
""",
    )
    write(
        root,
        "docs/profiles-and-risk-boundaries.md",
        "# Profiles and Risk Boundaries\n\n`omegaprompt` supports both with one engine.\n",
    )
    write(
        root,
        "src/omegaprompt/cli.py",
        '''
import typer
app = typer.Typer(name="omegaprompt")
app.command(name="calibrate")(lambda: None)
app.command(name="report")(lambda: None)
app.command(name="diff")(lambda: None)
app.command(name="check-artifact")(lambda: None)
'''.strip(),
    )
    write(
        root,
        "src/omegaprompt/commands/calibrate.py",
        '''
import typer
def calibrate():
    artifact = type("A", (), {"status": "OK"})()
    if artifact.status != "OK":
        raise typer.Exit(code=1)
'''.strip(),
    )
    write(root, "src/omegaprompt/commands/diff.py", "def diff():\n    pass\n")
    write(root, "src/omegaprompt/commands/report.py", "def report():\n    pass\n")
    write(
        root,
        "src/omegaprompt/runtime.py",
        '''
"""High-level entrypoints.

Tier 1: calibrate, evaluate, report, diff.
Tier 2: measure_sensitivity, grade, preflight, classify_traps.
"""
def calibrate(): pass
def evaluate(): pass
def report(): pass
def diff(): pass
def measure_sensitivity(): pass
def grade(): pass
def preflight(): pass
def classify_traps(): pass
'''.strip(),
    )
    write(
        root,
        "src/omegaprompt/mcp/server.py",
        '''
class App:
    def tool(self):
        def dec(fn): return fn
        return dec
mcp_app = App()
@mcp_app.tool()
def calibrate(): pass
@mcp_app.tool()
def evaluate(): pass
@mcp_app.tool()
def report(): pass
@mcp_app.tool()
def diff(): pass
@mcp_app.tool()
def measure_sensitivity(): pass
@mcp_app.tool()
def grade(): pass
@mcp_app.tool()
def preflight(): pass
@mcp_app.tool()
def classify_traps(): pass
'''.strip(),
    )
    write(root, "src/omegaprompt/mcp/__main__.py", "def main():\n    return 0\n")
    write(root, "src/omegaprompt/providers/gemini_provider.py", "class GeminiProvider:\n    placeholder=False\n")
    write(root, "src/omegaprompt/domain/result.py", 'class CalibrationArtifact:\n    schema_version: str = "2.0"\n')
    write_json(
        root,
        "examples/reference/reference_artifact.json",
        {
            "schema_version": "2.0",
            "status": "OK",
            "ship_recommendation": "ship",
            "target_capabilities": {"provider": "gemini", "placeholder": False},
            "judge_capabilities": {"provider": "openai", "placeholder": False},
            "walk_forward": {
                "train_best_fitness": 0.9,
                "test_fitness": 0.88,
                "generalization_gap": 0.02,
                "gap_status": "OK",
                "validation_mode": "auto",
                "shared_item_count": 4,
                "kc4_correlation": 0.8,
                "kc4_status": "COMPUTED",
                "max_gap_threshold": 0.25,
                "min_kc4_threshold": 0.5,
                "passed": True,
            },
        },
    )
    write(root, "examples/reference/reproduce_reference_artifact.py", "print('ok')\n")
    write(root, "examples/reference/reproduce_preflight_demo.py", "print('ok')\n")
    write_json(root, "examples/reference/reference_preflight_report.json", {"status": "proceed"})
    write_json(root, "examples/reference/reference_adaptation_plan.json", {"preserves_discipline": True})
    write(root, ".github/workflows/ci.yml", 'name: ci\njobs:\n  test:\n    steps:\n      - run: python -m pytest -q -m "not live"\n')
    write(root, "tests/test_smoke.py", "def test_smoke():\n    assert True\n")


def check_ids(report: dict) -> set[str]:
    return {item["id"] for item in report["checks"] if item["status"] != "OK"}


def test_checker_accepts_consistent_fixture(tmp_path: Path) -> None:
    make_consistent_repo(tmp_path)
    report = checker.run_checks(tmp_path)
    assert report["summary"]["strict_blocking_count"] == 0
    assert check_ids(report) == set()


def test_checker_detects_known_drift_fixture(tmp_path: Path) -> None:
    make_consistent_repo(tmp_path)
    pyproject = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    # Introduce a pyproject-vs-__init__ version mismatch from the consistent
    # base so __VERSION_MATCH drifts (the fixture base now uses 2.0.2 for both).
    write(tmp_path, "pyproject.toml", pyproject.replace('version = "2.0.2"', 'version = "2.0.3"'))
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    readme = readme.replace("v2.0.2", "v1.6.0")
    readme = readme.replace(
        "Exit codes: `0` when the artifact status is OK, `1` when the artifact status is not OK, `2` on argument or environment problems.",
        "Exit codes: `0` on success (regardless of `status`), `2` on environment problems.",
    )
    readme += "\nThe `omegaprompt` CLI binary remains as a compatibility alias during migration.\n"
    readme += "\nThe current head of `main` passes **149 tests**.\n"
    write(tmp_path, "README.md", readme)
    write(tmp_path, "docs/profiles-and-risk-boundaries.md", "`omegacal` supports both with one engine.\n")
    write(tmp_path, "docs/provider-capabilities.md", "| Gemini | Tier 2 cloud-grade | Placeholder | Not ship-grade | Adapter reserved |\n\n- explicit placeholder only\n")
    runtime = (tmp_path / "src/omegaprompt/runtime.py").read_text(encoding="utf-8")
    write(tmp_path, "src/omegaprompt/runtime.py", runtime.replace("Tier 2: measure_sensitivity", "Tier 2 (forthcoming): measure_sensitivity"))
    write_json(
        tmp_path,
        "examples/reference/reference_artifact.json",
        {
            "schema_version": "2.0",
            "status": "OK",
            "ship_recommendation": "hold",
            "target_capabilities": None,
            "judge_capabilities": None,
            "walk_forward": {
                "train_best_fitness": 0.9,
                "test_fitness": 0.9,
                "generalization_gap": 0.0,
                "kc4_correlation": None,
                "passed": True,
            },
        },
    )

    ids = check_ids(checker.run_checks(tmp_path))
    assert "__VERSION_MATCH" in ids
    assert "README_MD_TEST_COUNT_PROSE_DRIFT" in ids
    assert "README_MD_VERSION_CALLOUT_DRIFT" in ids
    assert "README_CALIBRATE_EXIT_CODE_DRIFT" in ids
    assert "OMEGACAL_PRIMARY_NAME_DRIFT" in ids
    assert "README_CLI_ALIAS_NAME_DRIFT" in ids
    assert "PROVIDER_DOC_GEMINI_PLACEHOLDER_DRIFT" in ids
    assert "REFERENCE_ARTIFACT_STATUS_SHIP_DRIFT" in ids
    assert "REFERENCE_ARTIFACT_CAPABILITY_NULLABILITY_DRIFT" in ids
    assert "REFERENCE_ARTIFACT_LEGACY_WALK_FORWARD_SHAPE" in ids
    assert "RUNTIME_TIER2_FORTHCOMING_DOCSTRING_DRIFT" in ids


def test_strict_json_output_reports_nonzero_for_drift(tmp_path: Path) -> None:
    make_consistent_repo(tmp_path)
    write(tmp_path, "docs/profiles-and-risk-boundaries.md", "`omegacal` supports both with one engine.\n")
    out = io.StringIO()
    rc = checker.main(
        ["--strict", "--json-output", "build/repo_consistency.json"],
        root=tmp_path,
        stdout=out,
    )
    assert rc == 1
    report_path = tmp_path / "build" / "repo_consistency.json"
    assert report_path.exists()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["summary"]["strict_blocking_count"] == 1
    assert "OMEGACAL_PRIMARY_NAME_DRIFT" in check_ids(data)
    assert "strict blocking findings: 1" in out.getvalue()


def test_readme_badge_composition_is_guarded(tmp_path: Path) -> None:
    make_consistent_repo(tmp_path)
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    readme = readme.replace("[![PyPI]", "[![Extra](x)](x)\n[![PyPI]")
    write(tmp_path, "README.md", readme)
    ids = check_ids(checker.run_checks(tmp_path))
    assert "README_BADGE_COMPOSITION" in ids


def test_file_error_classification() -> None:
    assert checker.classify_file_error(FileNotFoundError()) == "MISSING_FILE"
    assert checker.classify_file_error(PermissionError()) == "ENVIRONMENT_BLOCKED"
    assert checker.classify_file_error(OSError()) == "ENVIRONMENT_BLOCKED"


def test_workflow_publish_check_is_publish_precise(tmp_path: Path) -> None:
    """WORKFLOW_RELEASE_OR_PUBLISH_COMMANDS must flag REAL publish vectors, not
    the bare token 'pypi'. A scheduled, read-only consumer canary (the docking
    hardlock) that INSTALLS from PyPI and mentions 'PyPI' in comments is not a
    publish workflow and must pass; a real `twine upload` must still be caught.
    Regression guard for the bare-'pypi'-substring false positive that flagged
    the literal 'PyPI' in the canary's comments."""
    make_consistent_repo(tmp_path)

    # Docking consumer canary: installs deps from PyPI + omega-lock@main,
    # read-only permissions, no publish/tag/release command. Mentions "PyPI" in
    # comments (the exact thing the old bare-"pypi" heuristic false-flagged).
    write(
        tmp_path,
        ".github/workflows/omega-lock-compat.yml",
        '''name: omega-lock compat (consumer canary)
# Installs the pinned PyPI build, then force-reinstalls omega-lock@main so the
# consumer contract runs against PyPI vs @main. Read-only; publishes nothing.
on:
  schedule:
    - cron: "17 6 * * *"
  workflow_dispatch:
permissions:
  contents: read
jobs:
  compat:
    steps:
      - run: pip install -e ".[dev,mcp]"
      - run: pip install --force-reinstall --no-deps "omega-lock @ git+https://example/omega-lock@main"
      - run: python -m pytest -q tests/test_omega_lock_contract.py
''',
    )
    assert "WORKFLOW_RELEASE_OR_PUBLISH_COMMANDS" not in check_ids(checker.run_checks(tmp_path))


@pytest.mark.parametrize(
    "step_line",
    [
        "run: twine upload dist/*",
        "uses: pypa/gh-action-pypi-publish@release/v1",
        "run: uv publish",
        "run: poetry publish",
        "run: flit publish",
        "run: hatch publish",
        "run: gh release create v1.0.0",
        "uses: softprops/action-gh-release@v2",
        "uses: ncipollo/release-action@v1",
        "uses: actions/create-release@v1",
        "run: git tag v1.0.0",
        "run: git push --tags",
    ],
)
def test_workflow_publish_check_flags_real_vectors(tmp_path: Path, step_line: str) -> None:
    """Every real publish / release / tag VECTOR must trip the guard, so the
    bare-'pypi' precision fix did not weaken real-publish coverage."""
    make_consistent_repo(tmp_path)
    write(
        tmp_path,
        ".github/workflows/publish.yml",
        f"name: publish\njobs:\n  release:\n    steps:\n      - {step_line}\n",
    )
    assert "WORKFLOW_RELEASE_OR_PUBLISH_COMMANDS" in check_ids(checker.run_checks(tmp_path))


def _publish_vector_workflow(on_block: str) -> str:
    """A workflow carrying a real publish vector (twine upload) under on_block."""
    return f"name: publish\n{on_block}\njobs:\n  release:\n    steps:\n      - run: twine upload dist/*\n"


def test_release_gated_publish_workflow_is_tolerated(tmp_path: Path) -> None:
    """The canonical opt-in publish.yml (triggers limited to release +
    workflow_dispatch, PyPI trusted publishing) is the intended publish path --
    its publish vector must NOT be flagged."""
    make_consistent_repo(tmp_path)
    write(
        tmp_path,
        ".github/workflows/publish.yml",
        _publish_vector_workflow("on:\n  release:\n    types: [published]\n  workflow_dispatch:"),
    )
    assert "WORKFLOW_RELEASE_OR_PUBLISH_COMMANDS" not in check_ids(checker.run_checks(tmp_path))


@pytest.mark.parametrize(
    "on_block",
    [
        # indirect triggers that fire on push/PR -- must NOT be read as release-gated
        'on:\n  release:\n    types: [published]\n  workflow_run:\n    workflows: ["ci"]\n    types: [completed]',
        "on:\n  release:\n    types: [published]\n  repository_dispatch:",
        "on:\n  release:\n    types: [published]\n  create:",
        # direct default-CI triggers paired with release
        "on: [push, release]",
        "on:\n  push:\n    branches: [main]\n  release:\n    types: [published]",
    ],
)
def test_publish_vector_with_non_release_trigger_is_flagged(tmp_path: Path, on_block: str) -> None:
    """A publish vector paired with ANY trigger outside {release, workflow_dispatch}
    -- including indirect ones (workflow_run / repository_dispatch / create) that
    fire on push -- must STILL be flagged. The release-gated allowance must not
    silently widen by dropping unknown triggers."""
    make_consistent_repo(tmp_path)
    write(tmp_path, ".github/workflows/publish.yml", _publish_vector_workflow(on_block))
    assert "WORKFLOW_RELEASE_OR_PUBLISH_COMMANDS" in check_ids(checker.run_checks(tmp_path))
