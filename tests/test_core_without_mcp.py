"""Core import and CLI behavior when the MCP optional extra is absent."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_with_blocked_mcp(code: str) -> subprocess.CompletedProcess[str]:
    blocker = """
import importlib.abc
import sys

class BlockMcp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mcp" or fullname.startswith("mcp."):
            raise ModuleNotFoundError("No module named 'mcp'", name="mcp")
        return None

sys.meta_path.insert(0, BlockMcp())
"""
    return subprocess.run(
        [sys.executable, "-c", blocker + "\n" + code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_core_import_and_core_cli_do_not_import_mcp_sdk() -> None:
    proc = _run_with_blocked_mcp(
        """
import omegaprompt
import omegacal
from omegaprompt.cli import app as omegaprompt_app
from omegacal.cli import app as omegacal_app
assert omegaprompt.__version__
assert omegaprompt_app is not None
assert omegacal_app is not None
print("core-ok")
"""
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "core-ok"


def test_omegaprompt_mcp_reports_tooling_missing_without_mcp_extra() -> None:
    proc = _run_with_blocked_mcp(
        """
from omegaprompt.mcp.__main__ import main
raise SystemExit(main([]))
"""
    )

    assert proc.returncode == 2
    assert "TOOLING_MISSING" in proc.stderr
    assert "omegaprompt[mcp]" in proc.stderr
    assert "CalibrationArtifact" not in proc.stderr


def test_mcp_optional_extra_boundary_is_documented_in_packaging_and_readme() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    optional = pyproject["project"]["optional-dependencies"]
    scripts = pyproject["project"]["scripts"]
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    readme_kr = (REPO_ROOT / "README_KR.md").read_text(encoding="utf-8")

    assert not any(dep == "mcp" or dep.startswith("mcp>") for dep in dependencies)
    assert optional["mcp"] == ["mcp>=1.0.0"]
    assert scripts["omegaprompt-mcp"] == "omegaprompt.mcp.__main__:main"
    assert 'pip install "omegaprompt[mcp]"' in readme
    assert 'pip install "omegaprompt[mcp]"' in readme_kr
