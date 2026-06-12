#!/usr/bin/env python
"""Offline repository consistency checker for omegaprompt.

The checker intentionally reads only the local checkout. It does not import
provider SDKs, contact GitHub/PyPI, run live API calls, or modify files except
for the explicit --json-output path.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.11+ ships tomllib.
    tomllib = None  # type: ignore[assignment]


EXPECTED = {
    "github_repo": "hibou04-ops/omegaprompt",
    "distribution": "omegaprompt",
    "primary_package": "omegaprompt",
    "compat_package": "omegacal",
    "cli_executables": {
        "omegaprompt": "omegaprompt.cli:app",
        "omegacal": "omegacal.cli:app",
        "omegaprompt-mcp": "omegaprompt.mcp.__main__:main",
    },
    "optional_extra": "mcp",
    "artifact_schema_version": "2.0",
    "mcp_tools": {
        "calibrate",
        "evaluate",
        "report",
        "diff",
        "measure_sensitivity",
        "grade",
        "preflight",
        "classify_traps",
    },
    "runtime_entrypoints": {
        "calibrate",
        "evaluate",
        "report",
        "diff",
        "measure_sensitivity",
        "grade",
        "preflight",
        "classify_traps",
    },
    "cli_commands": {"calibrate", "report", "diff", "check-artifact", "gate"},
    "wheel_packages": {"src/omegaprompt", "src/omegacal"},
}

# Badge-composition tokens are version-AGNOSTIC on purpose: the PyPI badge's
# exact version is owned by the separate README_PYPI_BADGE_VERSION check
# (which compares the badge version to pyproject's source-of-truth version).
# Encoding the literal version here too created a coupling where every
# release bump broke README_BADGE_COMPOSITION even though the badge was
# correct. The composition check only needs a stable structural token to
# confirm "this is the PyPI badge" — ``pypi-`` does that for any version.
README_BADGES = [
    ("CI", "actions/workflows/ci.yml/badge.svg"),
    ("License: Apache 2.0", "license-Apache--2.0-blue.svg"),
    ("Python", "python-3.11%2B-blue.svg"),
    ("PyPI", "pypi-"),
    ("Tests", "tests-passing-brightgreen.svg"),
    ("Artifact schema", "artifact-schema%20v2.0-blueviolet.svg"),
    ("MCP", "MCP-server-blueviolet.svg"),
    ("Parent framework", "framework-omega--lock-blueviolet.svg"),
]


@dataclass
class Check:
    id: str
    status: str
    severity: str
    category: str
    message: str
    path: str | None = None
    expected: Any = None
    actual: Any = None
    remediation: str | None = None
    approved: bool = False

    @property
    def blocks_strict(self) -> bool:
        if self.approved:
            return False
        return self.status in {"DRIFT", "MISSING_FILE", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED"}

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "path": self.path,
            "expected": self.expected,
            "actual": self.actual,
            "remediation": self.remediation,
            "approved": self.approved,
        }


class Context:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.checks: list[Check] = []
        self.text_cache: dict[str, str] = {}

    def add(
        self,
        id: str,
        *,
        status: str,
        severity: str,
        category: str,
        message: str,
        path: str | None = None,
        expected: Any = None,
        actual: Any = None,
        remediation: str | None = None,
        approved: bool = False,
    ) -> None:
        self.checks.append(
            Check(
                id=id,
                status=status,
                severity=severity,
                category=category,
                message=message,
                path=path,
                expected=expected,
                actual=actual,
                remediation=remediation,
                approved=approved,
            )
        )

    def ok(
        self,
        id: str,
        message: str,
        *,
        category: str,
        path: str | None = None,
        expected: Any = None,
        actual: Any = None,
    ) -> None:
        self.add(
            id,
            status="OK",
            severity="INFO",
            category=category,
            message=message,
            path=path,
            expected=expected,
            actual=actual,
        )

    def drift(
        self,
        id: str,
        message: str,
        *,
        category: str,
        path: str | None = None,
        expected: Any = None,
        actual: Any = None,
        remediation: str | None = None,
        severity: str = "ERROR",
    ) -> None:
        self.add(
            id,
            status="DRIFT",
            severity=severity,
            category=category,
            message=message,
            path=path,
            expected=expected,
            actual=actual,
            remediation=remediation,
        )

    def missing_file(self, rel: str, message: str, *, category: str) -> None:
        self.add(
            f"MISSING_FILE:{rel}",
            status="MISSING_FILE",
            severity="ERROR",
            category=category,
            path=rel,
            message=message,
            remediation="Restore the required repository file or update the checker if the surface intentionally moved.",
        )

    def environment_blocked(self, rel: str, exc: BaseException, *, category: str) -> None:
        self.add(
            f"ENVIRONMENT_BLOCKED:{rel}",
            status="ENVIRONMENT_BLOCKED",
            severity="ERROR",
            category=category,
            path=rel,
            message=f"Cannot read required file: {exc}",
            remediation="Fix local filesystem permissions and rerun; this is not release approval.",
        )

    def tooling_missing(self, tool: str, message: str, *, category: str) -> None:
        self.add(
            f"TOOLING_MISSING:{tool}",
            status="TOOLING_MISSING",
            severity="ERROR",
            category=category,
            message=message,
            remediation="Install the missing local tooling or run with the supported Python version; this is not release approval.",
        )

    def read_text(self, rel: str, *, required: bool = True, category: str = "files") -> str | None:
        if rel in self.text_cache:
            return self.text_cache[rel]
        path = self.root / rel
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            if required:
                self.missing_file(rel, "Required consistency surface is absent.", category=category)
            return None
        except PermissionError as exc:
            self.environment_blocked(rel, exc, category=category)
            return None
        except OSError as exc:
            self.environment_blocked(rel, exc, category=category)
            return None
        self.text_cache[rel] = text
        return text

    def read_json(self, rel: str, *, required: bool = True, category: str = "files") -> Any | None:
        text = self.read_text(rel, required=required, category=category)
        if text is None:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            self.drift(
                f"INVALID_JSON:{rel}",
                f"JSON parse failed: {exc}",
                category=category,
                path=f"{rel}:{exc.lineno}",
                remediation="Fix the JSON syntax; the checker does not repair artifacts.",
            )
            return None


def classify_file_error(exc: BaseException) -> str:
    """Classify local file access failures for tests and callers."""
    if isinstance(exc, FileNotFoundError):
        return "MISSING_FILE"
    if isinstance(exc, (PermissionError, OSError)):
        return "ENVIRONMENT_BLOCKED"
    return "DRIFT"


def _parse_ast(text: str, rel: str, ctx: Context, category: str) -> ast.Module | None:
    try:
        return ast.parse(text)
    except SyntaxError as exc:
        ctx.drift(
            f"PYTHON_SYNTAX:{rel}",
            f"Python parse failed: {exc}",
            category=category,
            path=f"{rel}:{exc.lineno or 1}",
            remediation="Fix syntax before relying on consistency analysis.",
        )
        return None


def _literal_assign(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    try:
                        return ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        return None
    return None


def _line_number(text: str, needle: str) -> int | None:
    for idx, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return idx
    return None


def _line_ref(rel: str, text: str, needle: str) -> str:
    line = _line_number(text, needle)
    return rel if line is None else f"{rel}:{line}"


def _first_semver_heading(text: str) -> str | None:
    match = re.search(r"^## \[(\d+\.\d+\.\d+)\]", text, flags=re.MULTILINE)
    return match.group(1) if match else None


def _extract_init_version(text: str, ctx: Context) -> tuple[str | None, list[str]]:
    tree = _parse_ast(text, "src/omegaprompt/__init__.py", ctx, "package")
    if tree is None:
        return None, []
    version = _literal_assign(tree, "__version__")
    all_names = _literal_assign(tree, "__all__")
    if not isinstance(all_names, list):
        all_names = []
    return version if isinstance(version, str) else None, [str(x) for x in all_names]


def _function_defs(text: str, rel: str, ctx: Context, category: str) -> set[str]:
    tree = _parse_ast(text, rel, ctx, category)
    if tree is None:
        return set()
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


def _mcp_tool_defs(text: str, ctx: Context) -> set[str]:
    tree = _parse_ast(text, "src/omegaprompt/mcp/server.py", ctx, "mcp")
    if tree is None:
        return set()
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "tool"
                and isinstance(target.value, ast.Name)
                and target.value.id == "mcp_app"
            ):
                names.add(node.name)
    return names


def _cli_command_names(text: str) -> set[str]:
    return set(re.findall(r"app\.command\(\s*name=[\"']([^\"']+)[\"']", text))


def _read_pyproject(ctx: Context) -> dict[str, Any]:
    if tomllib is None:
        ctx.tooling_missing(
            "tomllib",
            "Python tomllib is unavailable; use Python 3.11+.",
            category="tooling",
        )
        return {}
    text = ctx.read_text("pyproject.toml", category="pyproject")
    if text is None:
        return {}
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:  # type: ignore[union-attr]
        ctx.drift(
            "PYPROJECT_TOML_PARSE",
            f"pyproject.toml parse failed: {exc}",
            category="pyproject",
            path="pyproject.toml",
            remediation="Fix TOML syntax before relying on package metadata.",
        )
        return {}


def check_pyproject(ctx: Context, pyproject: dict[str, Any]) -> dict[str, Any]:
    project = pyproject.get("project", {})
    build = pyproject.get("tool", {}).get("hatch", {}).get("build", {})
    wheel = build.get("targets", {}).get("wheel", {})
    project_urls = project.get("urls", {})

    name = project.get("name")
    version = project.get("version")
    dependencies = list(project.get("dependencies", []) or [])
    optional = project.get("optional-dependencies", {}) or {}
    scripts = project.get("scripts", {}) or {}
    packages = set(wheel.get("packages", []) or [])

    identities = {
        "github_repo": EXPECTED["github_repo"],
        "distribution": name,
        "primary_import_package": EXPECTED["primary_package"],
        "compat_import_package": EXPECTED["compat_package"],
        "cli_executables": sorted(scripts),
        "optional_extras": sorted(optional),
        "artifact_schema_version": EXPECTED["artifact_schema_version"],
        "mcp_tool_names": sorted(EXPECTED["mcp_tools"]),
    }

    if name == EXPECTED["distribution"]:
        ctx.ok("PYPROJECT_NAME", "PyPI distribution name is omegaprompt.", category="pyproject", path="pyproject.toml", expected=EXPECTED["distribution"], actual=name)
    else:
        ctx.drift("PYPROJECT_NAME", "PyPI distribution name drifted.", category="pyproject", path="pyproject.toml", expected=EXPECTED["distribution"], actual=name)

    repo_url = str(project_urls.get("Repository", ""))
    if EXPECTED["github_repo"] in repo_url:
        ctx.ok("PYPROJECT_REPOSITORY_URL", "Repository URL points at hibou04-ops/omegaprompt.", category="pyproject", path="pyproject.toml", expected=EXPECTED["github_repo"], actual=repo_url)
    else:
        ctx.drift("PYPROJECT_REPOSITORY_URL", "Repository URL does not identify the expected GitHub repo.", category="pyproject", path="pyproject.toml", expected=EXPECTED["github_repo"], actual=repo_url)

    expected_scripts = EXPECTED["cli_executables"]
    if scripts == expected_scripts:
        ctx.ok("PYPROJECT_SCRIPTS", "CLI executables map to the expected entrypoints.", category="pyproject", path="pyproject.toml", expected=expected_scripts, actual=scripts)
    else:
        ctx.drift("PYPROJECT_SCRIPTS", "CLI executable mapping drifted.", category="pyproject", path="pyproject.toml", expected=expected_scripts, actual=scripts)

    if packages == EXPECTED["wheel_packages"]:
        ctx.ok("PYPROJECT_WHEEL_PACKAGES", "Wheel includes primary and compatibility packages.", category="pyproject", path="pyproject.toml", expected=sorted(EXPECTED["wheel_packages"]), actual=sorted(packages))
    else:
        ctx.drift("PYPROJECT_WHEEL_PACKAGES", "Wheel package list does not preserve omegaprompt + omegacal.", category="pyproject", path="pyproject.toml", expected=sorted(EXPECTED["wheel_packages"]), actual=sorted(packages))

    dep_names = {_dependency_name(d) for d in dependencies}
    expected_deps = {"typer", "pydantic", "omega-lock", "anthropic", "openai", "google-genai"}
    missing = sorted(expected_deps - dep_names)
    if not missing:
        ctx.ok("PYPROJECT_DEPENDENCIES", "Core dependencies include provider adapters and calibration engine.", category="pyproject", path="pyproject.toml", expected=sorted(expected_deps), actual=sorted(dep_names))
    else:
        ctx.drift("PYPROJECT_DEPENDENCIES", "Core dependency set is missing expected packages.", category="pyproject", path="pyproject.toml", expected=sorted(expected_deps), actual=sorted(dep_names), remediation="Restore missing dependencies without moving MCP into default dependencies.")

    if "mcp" not in dep_names and "mcp" in optional and any(_dependency_name(d) == "mcp" for d in optional.get("mcp", [])):
        ctx.ok("MCP_OPTIONAL_EXTRA_BOUNDARY", "MCP remains an optional extra, not a default dependency.", category="pyproject", path="pyproject.toml", expected='omegaprompt[mcp]', actual={"default_has_mcp": "mcp" in dep_names, "mcp_extra": optional.get("mcp")})
    else:
        ctx.drift("MCP_OPTIONAL_EXTRA_BOUNDARY", "MCP dependency boundary drifted.", category="pyproject", path="pyproject.toml", expected="mcp only under [project.optional-dependencies].mcp", actual={"dependencies": dependencies, "mcp_extra": optional.get("mcp")}, remediation="Keep MCP installable as omegaprompt[mcp] and out of default dependencies.")

    return {"version": version, "identities": identities}


def _dependency_name(spec: str) -> str:
    return re.split(r"[<>=!~;\[]", spec, maxsplit=1)[0].strip().lower()


def check_package_surface(ctx: Context, py_version: str | None) -> None:
    init_text = ctx.read_text("src/omegaprompt/__init__.py", category="package")
    if init_text is None:
        return
    init_version, all_names = _extract_init_version(init_text, ctx)
    if init_version == py_version:
        ctx.ok("__VERSION_MATCH", "__version__ matches pyproject.toml.", category="package", path="src/omegaprompt/__init__.py", expected=py_version, actual=init_version)
    else:
        ctx.drift("__VERSION_MATCH", "pyproject version and package __version__ differ.", category="package", path="src/omegaprompt/__init__.py", expected=py_version, actual=init_version, remediation="Bump pyproject.toml and src/omegaprompt/__init__.py together.")

    missing_exports = sorted(EXPECTED["runtime_entrypoints"] - set(all_names))
    if not missing_exports:
        ctx.ok("PACKAGE_RUNTIME_EXPORTS", "Package __all__ exposes all runtime entrypoints.", category="package", path="src/omegaprompt/__init__.py", expected=sorted(EXPECTED["runtime_entrypoints"]), actual=sorted(EXPECTED["runtime_entrypoints"] & set(all_names)))
    else:
        ctx.drift("PACKAGE_RUNTIME_EXPORTS", "Package __all__ misses runtime entrypoints.", category="package", path="src/omegaprompt/__init__.py", expected=sorted(EXPECTED["runtime_entrypoints"]), actual=sorted(set(all_names)), remediation="Expose runtime entrypoints consistently at package level.")

    omegacal_init = ctx.read_text("src/omegacal/__init__.py", category="package")
    omegacal_cli = ctx.read_text("src/omegacal/cli.py", category="package")
    omegacal_main = ctx.read_text("src/omegacal/__main__.py", category="package")
    alias_ok = (
        omegacal_init is not None
        and "from omegaprompt import *" in omegacal_init
        and omegacal_cli is not None
        and "from omegaprompt.cli import app" in omegacal_cli
        and omegacal_main is not None
        and "from omegaprompt.cli import app" in omegacal_main
    )
    if alias_ok:
        ctx.ok("OMEGACAL_ALIAS_PACKAGE", "omegacal remains a thin compatibility alias over omegaprompt.", category="package", path="src/omegacal")
    else:
        ctx.drift("OMEGACAL_ALIAS_PACKAGE", "omegacal compatibility alias drifted.", category="package", path="src/omegacal", expected="thin imports from omegaprompt", actual="see src/omegacal files", remediation="Keep omegacal as compatibility surface; do not rename the primary package.")


def check_readme_badges_and_versions(ctx: Context, py_version: str | None) -> None:
    readme = ctx.read_text("README.md", category="docs")
    if readme is None:
        return

    badge_lines = [line.strip() for line in readme.splitlines()[:20] if line.strip().startswith("[![")]
    expected_lines = [
        line
        for line in (
            f"[![{label}]" for label, _ in README_BADGES
        )
    ]
    actual_labels = []
    for line in badge_lines:
        match = re.match(r"\[\!\[([^\]]+)\]", line)
        actual_labels.append(match.group(1) if match else line)
    expected_labels = [label for label, _ in README_BADGES]
    substrings_ok = len(badge_lines) == len(README_BADGES) and all(
        label == actual_label and token in line
        for (label, token), actual_label, line in zip(README_BADGES, actual_labels, badge_lines)
    )
    if substrings_ok:
        ctx.ok("README_BADGE_COMPOSITION", "README.md top badge row composition is preserved.", category="docs", path="README.md:5", expected=expected_labels, actual=actual_labels)
    else:
        ctx.drift("README_BADGE_COMPOSITION", "README.md badge row composition changed.", category="docs", path="README.md:5", expected=README_BADGES, actual=badge_lines, remediation="Do not add, remove, reorder, or restyle the current README.md badge row in this task.")

    badge_version = _extract_pypi_badge_version(readme)
    if badge_version == py_version:
        ctx.ok("README_PYPI_BADGE_VERSION", "README PyPI badge matches pyproject version.", category="docs", path="README.md", expected=py_version, actual=badge_version)
    else:
        ctx.drift("README_PYPI_BADGE_VERSION", "README PyPI badge version drifted from pyproject.", category="docs", path="README.md", expected=py_version, actual=badge_version)

    for rel in ("README.md", "README_KR.md"):
        text = ctx.read_text(rel, category="docs")
        if text is None:
            continue
        badge_count = _extract_test_badge_count(text)
        prose_claims = _extract_readme_test_count_claims(text)
        if badge_count is None:
            # The test badge is intentionally count-free (e.g. tests-passing), so
            # it can never go stale. With no badge count to anchor against, any
            # exact prose test count is itself unanchored and will re-stale on the
            # next release, so it is disallowed outright.
            mismatches = prose_claims
        else:
            mismatches = [claim for claim in prose_claims if claim["count"] != badge_count]
        if mismatches:
            ctx.drift(
                f"{rel.upper().replace('.', '_')}_TEST_COUNT_PROSE_DRIFT",
                (
                    f"{rel} has exact prose test-count claims but the test badge declares no count."
                    if badge_count is None
                    else f"{rel} has exact prose test-count claims that disagree with its badge."
                ),
                category="docs",
                path=rel,
                expected={"badge_tests_passing": badge_count},
                actual=mismatches,
                remediation=(
                    "Remove exact prose test-count claims; the test badge is intentionally count-free so prose counts cannot be anchored."
                    if badge_count is None
                    else "Update prose or remove exact current test-count claims; do not alter README.md badge composition in this task."
                ),
                severity="WARNING",
            )
        else:
            ctx.ok(f"{rel.upper().replace('.', '_')}_TEST_COUNT_PROSE", f"{rel} has no prose test-count drift against its badge.", category="docs", path=rel, expected=badge_count, actual=prose_claims)

        callouts = _extract_top_version_callouts(text)
        stale = [c for c in callouts if c["version"] != py_version]
        if stale:
            ctx.drift(
                f"{rel.upper().replace('.', '_')}_VERSION_CALLOUT_DRIFT",
                f"{rel} top release callout does not match pyproject version.",
                category="docs",
                path=rel,
                expected=py_version,
                actual=stale,
                remediation="Either make the callout historical explicitly or update it with the current source-of-truth version.",
                severity="WARNING",
            )
        else:
            ctx.ok(f"{rel.upper().replace('.', '_')}_VERSION_CALLOUT", f"{rel} top release callout matches the package version or is absent.", category="docs", path=rel, expected=py_version, actual=callouts)

    changelog = ctx.read_text("CHANGELOG.md", category="docs")
    if changelog is not None:
        latest = _first_semver_heading(changelog)
        if latest == py_version:
            ctx.ok("CHANGELOG_LATEST_VERSION", "CHANGELOG latest release heading matches pyproject version.", category="docs", path="CHANGELOG.md", expected=py_version, actual=latest)
        else:
            ctx.drift("CHANGELOG_LATEST_VERSION", "CHANGELOG latest release heading drifted from pyproject.", category="docs", path="CHANGELOG.md", expected=py_version, actual=latest, remediation="Keep CHANGELOG release heading aligned with package version.")


def _extract_pypi_badge_version(text: str) -> str | None:
    match = re.search(r"pypi-(\d+\.\d+\.\d+)-blue\.svg", text)
    return match.group(1) if match else None


def _extract_test_badge_count(text: str) -> int | None:
    match = re.search(r"tests-(\d+)%20passing", text)
    return int(match.group(1)) if match else None


def _extract_readme_test_count_claims(text: str) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        lower = line.lower()
        if "badge" in lower:
            continue
        external_project_line = "omega-lock" in lower or "antemortem" in lower
        current_repo_line = (
            "current head" in lower
            or "test suite" in lower
            or "ci" in lower
            or "live api call" in lower
        )
        if external_project_line and not current_repo_line:
            continue
        for match in re.finditer(r"(?<![\d.])(\d{2,4})\s+tests?\b", line, flags=re.IGNORECASE):
            claims.append({"line": idx, "count": int(match.group(1)), "text": line.strip()})
    return claims


def _extract_top_version_callouts(text: str) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    top = "\n".join(text.splitlines()[:40])
    for match in re.finditer(r"> \*\*v(\d+\.\d+\.\d+)\b", top):
        line = top[: match.start()].count("\n") + 1
        claims.append({"line": line, "version": match.group(1)})
    return claims


def check_docs_naming_and_capabilities(ctx: Context) -> None:
    docs = {
        rel: ctx.read_text(rel, category="docs")
        for rel in (
            "README.md",
            "README_KR.md",
            "EASY_README.md",
            "EASY_README_KR.md",
            "docs/provider-capabilities.md",
            "docs/profiles-and-risk-boundaries.md",
        )
    }

    profiles = docs.get("docs/profiles-and-risk-boundaries.md")
    if profiles is not None and "`omegacal` supports both with one engine" in profiles:
        ctx.drift(
            "OMEGACAL_PRIMARY_NAME_DRIFT",
            "profiles-and-risk-boundaries.md uses the compatibility alias as the primary product name.",
            category="docs",
            path=_line_ref("docs/profiles-and-risk-boundaries.md", profiles, "`omegacal` supports both"),
            expected="omegaprompt as primary name; omegacal only as compatibility alias",
            actual="`omegacal` supports both with one engine",
            remediation="Use omegaprompt for primary product docs and reserve omegacal for alias/migration notes.",
            severity="WARNING",
        )
    elif profiles is not None:
        ctx.ok("OMEGACAL_PRIMARY_NAME_DRIFT", "profiles-and-risk-boundaries.md uses primary naming consistently.", category="docs", path="docs/profiles-and-risk-boundaries.md")

    readme = docs.get("README.md")
    if readme is not None and "The `omegaprompt` CLI binary remains as a compatibility alias" in readme:
        ctx.drift(
            "README_CLI_ALIAS_NAME_DRIFT",
            "README describes the primary CLI as a compatibility alias.",
            category="docs",
            path=_line_ref("README.md", readme, "The `omegaprompt` CLI binary remains"),
            expected="omegacal is compatibility alias; omegaprompt is primary CLI",
            actual="omegaprompt described as compatibility alias",
            remediation="Keep the primary/alias distinction explicit.",
            severity="WARNING",
        )
    elif readme is not None:
        ctx.ok("README_CLI_ALIAS_NAME_DRIFT", "README primary CLI naming is consistent.", category="docs", path="README.md")

    provider_doc = docs.get("docs/provider-capabilities.md")
    gemini_code = ctx.read_text("src/omegaprompt/providers/gemini_provider.py", category="providers")
    if provider_doc is not None and gemini_code is not None:
        code_implemented = "class GeminiProvider" in gemini_code and "placeholder=False" in gemini_code
        doc_placeholder = bool(re.search(r"\|\s*Gemini\s*\|[^\n]*\|\s*Placeholder\s*\|", provider_doc)) or "explicit placeholder only" in provider_doc
        if code_implemented and doc_placeholder:
            ctx.drift(
                "PROVIDER_DOC_GEMINI_PLACEHOLDER_DRIFT",
                "provider-capabilities.md still describes Gemini as a placeholder, but the adapter reports placeholder=False.",
                category="providers",
                path=_line_ref("docs/provider-capabilities.md", provider_doc, "| Gemini |"),
                expected="Gemini implemented adapter with placeholder=False; still non-ship-grade judge unless validated",
                actual="Gemini documented as Placeholder / explicit placeholder only",
                remediation="Update capability docs to distinguish implemented target adapter from non-ship-grade judge status.",
            )
        else:
            ctx.ok("PROVIDER_DOC_GEMINI_PLACEHOLDER_DRIFT", "Gemini provider docs agree with code-level placeholder status.", category="providers", path="docs/provider-capabilities.md")


def check_cli_runtime_mcp(ctx: Context) -> None:
    cli_text = ctx.read_text("src/omegaprompt/cli.py", category="cli")
    calibrate_text = ctx.read_text("src/omegaprompt/commands/calibrate.py", category="cli")
    runtime_text = ctx.read_text("src/omegaprompt/runtime.py", category="runtime")
    mcp_text = ctx.read_text("src/omegaprompt/mcp/server.py", category="mcp")
    readme = ctx.read_text("README.md", category="docs")

    if cli_text is not None:
        command_names = _cli_command_names(cli_text)
        if command_names == EXPECTED["cli_commands"]:
            ctx.ok("CLI_COMMAND_SET", "Top-level Typer app wires the expected CLI commands.", category="cli", path="src/omegaprompt/cli.py", expected=sorted(EXPECTED["cli_commands"]), actual=sorted(command_names))
        else:
            ctx.drift("CLI_COMMAND_SET", "CLI command set drifted.", category="cli", path="src/omegaprompt/cli.py", expected=sorted(EXPECTED["cli_commands"]), actual=sorted(command_names), remediation="Keep CLI commands distinct from runtime-only/MCP-only entrypoints unless intentionally adding CLI support.")

    if readme is not None and calibrate_text is not None:
        actual_nonzero_on_non_ok = 'artifact.status != "OK"' in calibrate_text and "typer.Exit(code=1)" in calibrate_text
        claim_zero_regardless_status = bool(
            re.search(
                r"`?0`?\s+on\s+success\s+\(regardless\s+of\s+`?status`?\)",
                readme,
                flags=re.IGNORECASE,
            )
        )
        if actual_nonzero_on_non_ok and claim_zero_regardless_status:
            ctx.drift(
                "README_CALIBRATE_EXIT_CODE_DRIFT",
                "README says calibrate exits 0 regardless of artifact status, but CLI exits 1 when status is not OK.",
                category="cli",
                path=_line_ref("README.md", readme, "Exit codes: `0` on success"),
                expected="README reflects code: non-OK artifact status exits 1",
                actual="README claims 0 regardless of status",
                remediation="Document actual behavior or intentionally change CLI with tests.",
            )
        else:
            ctx.ok("README_CALIBRATE_EXIT_CODE_DRIFT", "README calibrate exit-code claim matches command behavior.", category="cli", path="README.md")

        if "calibrate / report / diff / preflight" in readme:
            ctx.drift(
                "README_PRELIGHT_CLI_COMMAND_DRIFT",
                "README mentions a preflight CLI contract, but the CLI command set does not include preflight.",
                category="cli",
                path=_line_ref("README.md", readme, "calibrate / report / diff / preflight"),
                expected=sorted(EXPECTED["cli_commands"]),
                actual="README mentions preflight CLI",
                remediation="Keep preflight described as Python/MCP runtime unless a CLI command is added with tests.",
                severity="WARNING",
            )

    if runtime_text is not None:
        defs = _function_defs(runtime_text, "src/omegaprompt/runtime.py", ctx, "runtime")
        missing = sorted(EXPECTED["runtime_entrypoints"] - defs)
        if not missing:
            ctx.ok("RUNTIME_ENTRYPOINT_SET", "runtime.py defines all eight expected entrypoints.", category="runtime", path="src/omegaprompt/runtime.py", expected=sorted(EXPECTED["runtime_entrypoints"]), actual=sorted(EXPECTED["runtime_entrypoints"] & defs))
        else:
            ctx.drift("RUNTIME_ENTRYPOINT_SET", "runtime.py is missing expected entrypoints.", category="runtime", path="src/omegaprompt/runtime.py", expected=sorted(EXPECTED["runtime_entrypoints"]), actual=sorted(defs), remediation="Keep runtime, MCP, package exports, and docs in sync.")

        if "Tier 2 (forthcoming)" in runtime_text and EXPECTED["runtime_entrypoints"].issubset(defs):
            ctx.drift(
                "RUNTIME_TIER2_FORTHCOMING_DOCSTRING_DRIFT",
                "runtime.py docstring says Tier 2 is forthcoming even though Tier 2 entrypoints exist.",
                category="runtime",
                path=_line_ref("src/omegaprompt/runtime.py", runtime_text, "Tier 2 (forthcoming)"),
                expected="Tier 2 entrypoints documented as shipped",
                actual="Tier 2 (forthcoming)",
                remediation="Update docstring; do not remove existing Tier 2 functions.",
                severity="WARNING",
            )
        else:
            ctx.ok("RUNTIME_TIER2_FORTHCOMING_DOCSTRING_DRIFT", "runtime.py Tier 2 docstring matches implemented entrypoints.", category="runtime", path="src/omegaprompt/runtime.py")

    if mcp_text is not None:
        mcp_tools = _mcp_tool_defs(mcp_text, ctx)
        if mcp_tools == EXPECTED["mcp_tools"]:
            ctx.ok("MCP_TOOL_SET", "MCP server exposes exactly the eight runtime tools.", category="mcp", path="src/omegaprompt/mcp/server.py", expected=sorted(EXPECTED["mcp_tools"]), actual=sorted(mcp_tools))
        else:
            ctx.drift("MCP_TOOL_SET", "MCP tool set drifted from runtime contract.", category="mcp", path="src/omegaprompt/mcp/server.py", expected=sorted(EXPECTED["mcp_tools"]), actual=sorted(mcp_tools), remediation="Keep MCP tool names distinct and synchronized with runtime entrypoints.")


def check_artifact_schema_and_reference(ctx: Context) -> None:
    result_text = ctx.read_text("src/omegaprompt/domain/result.py", category="artifact")
    if result_text is not None:
        schema_default = re.search(r"schema_version:\s*str\s*=\s*[\"']([^\"']+)[\"']", result_text)
        actual = schema_default.group(1) if schema_default else None
        if actual == EXPECTED["artifact_schema_version"]:
            ctx.ok("ARTIFACT_SCHEMA_DEFAULT", "CalibrationArtifact schema_version default is v2.0.", category="artifact", path="src/omegaprompt/domain/result.py", expected=EXPECTED["artifact_schema_version"], actual=actual)
        else:
            ctx.drift("ARTIFACT_SCHEMA_DEFAULT", "CalibrationArtifact schema_version default drifted.", category="artifact", path="src/omegaprompt/domain/result.py", expected=EXPECTED["artifact_schema_version"], actual=actual, remediation="Only bump schema with explicit migration and docs.")

    artifact = ctx.read_json("examples/reference/reference_artifact.json", category="examples")
    if not isinstance(artifact, dict):
        return

    if artifact.get("schema_version") == EXPECTED["artifact_schema_version"]:
        ctx.ok("REFERENCE_ARTIFACT_SCHEMA_VERSION", "Reference artifact schema_version matches v2.0.", category="examples", path="examples/reference/reference_artifact.json", expected=EXPECTED["artifact_schema_version"], actual=artifact.get("schema_version"))
    else:
        ctx.drift("REFERENCE_ARTIFACT_SCHEMA_VERSION", "Reference artifact schema_version drifted.", category="examples", path="examples/reference/reference_artifact.json", expected=EXPECTED["artifact_schema_version"], actual=artifact.get("schema_version"))

    status = artifact.get("status")
    ship = artifact.get("ship_recommendation")
    if status == "OK" and ship != "ship":
        ctx.drift(
            "REFERENCE_ARTIFACT_STATUS_SHIP_DRIFT",
            "Reference artifact status is OK but ship_recommendation is not ship.",
            category="examples",
            path="examples/reference/reference_artifact.json",
            expected={"status": "OK", "ship_recommendation": "ship"},
            actual={"status": status, "ship_recommendation": ship},
            remediation="Regenerate or intentionally annotate the reference artifact; do not silently rewrite it inside the checker.",
            severity="WARNING",
        )
    else:
        ctx.ok("REFERENCE_ARTIFACT_STATUS_SHIP_DRIFT", "Reference artifact status and ship recommendation are coherent.", category="examples", path="examples/reference/reference_artifact.json", expected="OK implies ship unless explicit risk state exists", actual={"status": status, "ship_recommendation": ship})

    null_caps = [k for k in ("target_capabilities", "judge_capabilities") if artifact.get(k) is None]
    if null_caps:
        ctx.drift(
            "REFERENCE_ARTIFACT_CAPABILITY_NULLABILITY_DRIFT",
            "Reference artifact leaves provider capability records null.",
            category="examples",
            path="examples/reference/reference_artifact.json",
            expected="target_capabilities and judge_capabilities populated or documented as intentionally null",
            actual=null_caps,
            remediation="Regenerate the reference artifact with capability records or document why null is intentional.",
            severity="WARNING",
        )
    else:
        ctx.ok("REFERENCE_ARTIFACT_CAPABILITY_NULLABILITY_DRIFT", "Reference artifact includes provider capability records.", category="examples", path="examples/reference/reference_artifact.json")

    walk = artifact.get("walk_forward")
    if isinstance(walk, dict):
        required = {
            "gap_status",
            "validation_mode",
            "shared_item_count",
            "kc4_status",
            "max_gap_threshold",
            "min_kc4_threshold",
        }
        missing = sorted(required - set(walk))
        if missing:
            ctx.drift(
                "REFERENCE_ARTIFACT_LEGACY_WALK_FORWARD_SHAPE",
                "Reference artifact walk_forward block has legacy shape and misses v2.0 explanatory fields.",
                category="examples",
                path="examples/reference/reference_artifact.json",
                expected=sorted(required),
                actual={"missing": missing, "present": sorted(walk)},
                remediation="Regenerate reference artifact with the current WalkForwardResult shape.",
                severity="WARNING",
            )
        else:
            ctx.ok("REFERENCE_ARTIFACT_LEGACY_WALK_FORWARD_SHAPE", "Reference artifact walk_forward block uses current explanatory fields.", category="examples", path="examples/reference/reference_artifact.json")

    for rel in (
        "examples/reference/reproduce_reference_artifact.py",
        "examples/reference/reproduce_preflight_demo.py",
        "examples/reference/reference_preflight_report.json",
        "examples/reference/reference_adaptation_plan.json",
    ):
        if (ctx.root / rel).exists():
            ctx.ok(f"REFERENCE_FILE_PRESENT:{rel}", "Reference support file exists.", category="examples", path=rel)
        else:
            ctx.missing_file(rel, "Reference support file is absent.", category="examples")


def _workflow_is_release_gated(text: str) -> bool:
    """True when a workflow's ``on:`` triggers are a SUBSET of the deliberate
    release surface (``release`` + ``workflow_dispatch``).

    This distinguishes the canonical, opt-in ``publish.yml`` (fires ONLY when a
    human cuts a GitHub Release or manually dispatches it, via PyPI trusted
    publishing) from a publish/tag/release command smuggled into a workflow that
    runs on every PR/push -- directly (``push``/``pull_request``/``schedule``) OR
    INDIRECTLY (``workflow_run`` after CI, ``repository_dispatch``, ``create``).
    A publish VECTOR is tolerated only inside a release-gated workflow; in any
    other workflow it is still flagged.

    Conservative by construction: ANY trigger outside {release, workflow_dispatch}
    -- including unrecognized ones -- disqualifies release-gating, so an
    accidental publish vector still trips the guard. A workflow with no
    recognizable ``on:`` block is NOT release-gated. We must NOT drop unknown
    triggers (a closed allowlist would silently widen the guard).
    """
    release_triggers = {"release", "workflow_dispatch"}

    lines = text.splitlines()
    keys: set[str] = set()
    # Inline form: `on: [release, workflow_dispatch]` or `on: push`.
    on_inline_seen = False
    for line in lines:
        inline = re.match(r"on:\s*\[([^\]]*)\]\s*$", line)
        if inline:
            on_inline_seen = True
            keys |= {k.strip() for k in inline.group(1).split(",") if k.strip()}
            break
        inline_map = re.match(r"on:\s*(\w+)\s*$", line)  # `on: push`
        if inline_map:
            on_inline_seen = True
            keys.add(inline_map.group(1))
            break
    if not on_inline_seen:
        # Block form: collect ONLY the top-level trigger keys -- those indented
        # one level directly under a bare `on:` line. The first such key sets the
        # trigger indent; deeper-nested keys (``types``/``branches``/``cron``/
        # ``workflows`` under a trigger) are NOT triggers and are skipped, so we
        # can classify on the real trigger set WITHOUT a vocabulary filter. The
        # scan stops at the next column-0 key, so job names under `jobs:` are out.
        in_on_block = False
        trigger_indent: int | None = None
        for line in lines:
            if re.match(r"on:\s*$", line):
                in_on_block = True
                continue
            if in_on_block:
                if re.match(r"\S", line):  # back to column 0 -> on-block ended
                    break
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                m = re.match(r"(\s+)(\w+):", line)
                if m:
                    indent = len(m.group(1))
                    if trigger_indent is None:
                        trigger_indent = indent
                    if indent == trigger_indent:
                        keys.add(m.group(2))

    # Do NOT drop unknown triggers: any trigger outside the release surface means
    # the workflow can fire outside a deliberate release and is not release-gated.
    if not keys:
        return False
    return keys <= release_triggers


def check_workflows_and_tests(ctx: Context) -> None:
    workflows = sorted((ctx.root / ".github" / "workflows").glob("*.yml"))
    workflows += sorted((ctx.root / ".github" / "workflows").glob("*.yaml"))
    if not workflows:
        ctx.missing_file(".github/workflows", "No GitHub workflow files found.", category="ci")
    workflow_texts: list[tuple[str, str]] = []
    combined = ""
    for path in workflows:
        rel = path.relative_to(ctx.root).as_posix()
        text = ctx.read_text(rel, category="ci")
        if text:
            workflow_texts.append((rel, text))
            combined += "\n" + text

    # Publish/release VECTORS only -- NOT the bare token "pypi", which would
    # false-match the literal "PyPI" in the prose/comments of a read-only
    # consumer canary that merely INSTALLS from PyPI -- consumption, not a
    # publish workflow (see .github/workflows/omega-lock-compat.yml). Real
    # vectors: twine, the pypa `pypi-publish` action, uv/poetry/flit/hatch
    # publish, the `gh release` CLI + action-gh-release/release-action,
    # create-release, `git tag`, and `git push --tags`. NOT bare "build"
    # (`python -m build` in wheel-smoke is legit) nor bare " publish"
    # ("publishes nothing" in comments). Coverage locked by
    # test_workflow_publish_check_flags_real_vectors; false-positive regression
    # by test_workflow_publish_check_is_publish_precise.
    #
    # A publish vector is TOLERATED only inside a release-gated workflow (the
    # canonical `publish.yml`: triggers limited to release + workflow_dispatch,
    # PyPI trusted publishing). The same vector in a default-CI workflow (one
    # that runs on pull_request / push / schedule) is still flagged -- that is
    # the accidental-publish class this guard exists for.
    forbidden = [
        "twine upload", "pypi-publish",
        "uv publish", "poetry publish", "flit publish", "hatch publish",
        "gh release", "gh-release", "release-action", "create-release",
        "git tag", "push --tags",
    ]
    hits: list[str] = []
    for rel, text in workflow_texts:
        if _workflow_is_release_gated(text):
            # Deliberate, opt-in release workflow; its publish vectors are the
            # intended path, not accidental-publish risk.
            continue
        for token in forbidden:
            if token.lower() in text.lower() and token not in hits:
                hits.append(token)
    if hits:
        ctx.drift(
            "WORKFLOW_RELEASE_OR_PUBLISH_COMMANDS",
            "A non-release-gated workflow contains publish/tag/release-like commands.",
            category="ci",
            path=".github/workflows",
            expected="default CI only runs offline tests; publish vectors live only in a release-gated workflow",
            actual=hits,
            remediation="Move publishing/release/tagging into a workflow triggered only by release + workflow_dispatch, or remove it from default CI.",
        )
    else:
        ctx.ok("WORKFLOW_RELEASE_OR_PUBLISH_COMMANDS", "No default-CI workflow publishes to PyPI, pushes tags, or creates releases; publish vectors are confined to the release-gated workflow.", category="ci", path=".github/workflows")

    live_excluding_pytest = (
        'python -m pytest -q -m "not live"' in combined
        or "python -m pytest -q -m 'not live'" in combined
    )
    if live_excluding_pytest:
        ctx.ok("WORKFLOW_DEFAULT_TEST_COMMAND", "Default CI runs pytest with the live marker excluded.", category="ci", path=".github/workflows", expected='python -m pytest -q -m "not live"', actual="present")
    else:
        ctx.drift("WORKFLOW_DEFAULT_TEST_COMMAND", "Default CI pytest command does not explicitly exclude live tests.", category="ci", path=".github/workflows", expected='python -m pytest -q -m "not live"', actual="absent", remediation="Keep default CI offline and deterministic by excluding the live marker.")

    test_files = sorted((ctx.root / "tests").glob("test_*.py"))
    if test_files:
        ctx.ok("TEST_DIRECTORY_PRESENT", "tests directory contains pytest files.", category="tests", path="tests", actual=len(test_files))
    else:
        ctx.missing_file("tests", "No test_*.py files found.", category="tests")

    suspicious_live = []
    for path in test_files:
        rel = path.relative_to(ctx.root).as_posix()
        text = ctx.read_text(rel, category="tests")
        if not text:
            continue
        if "OMEGAPROMPT_LIVE_PROVIDER_TESTS" in text:
            continue
        if re.search(r"\b(anthropic|openai)\.(Anthropic|OpenAI)\(", text) or "genai.Client(" in text:
            suspicious_live.append(rel)
    if suspicious_live:
        ctx.drift(
            "DEFAULT_TESTS_LIVE_PROVIDER_CALLS",
            "Potential live provider client construction found without OMEGAPROMPT_LIVE_PROVIDER_TESTS guard.",
            category="tests",
            path="tests",
            expected="live provider tests skipped by default behind explicit opt-in",
            actual=suspicious_live,
            remediation="Guard live provider tests with OMEGAPROMPT_LIVE_PROVIDER_TESTS=1 or replace with mocked clients.",
        )
    else:
        ctx.ok("DEFAULT_TESTS_LIVE_PROVIDER_CALLS", "Default tests do not appear to construct live provider clients.", category="tests", path="tests")


def run_checks(root: Path | str = ".") -> dict[str, Any]:
    ctx = Context(Path(root).resolve())
    pyproject = _read_pyproject(ctx)
    py_facts = check_pyproject(ctx, pyproject)
    py_version = py_facts.get("version")

    check_package_surface(ctx, py_version)
    check_readme_badges_and_versions(ctx, py_version)
    check_docs_naming_and_capabilities(ctx)
    check_cli_runtime_mcp(ctx)
    check_artifact_schema_and_reference(ctx)
    check_workflows_and_tests(ctx)

    status_counts: dict[str, int] = {}
    for check in ctx.checks:
        status_counts[check.status] = status_counts.get(check.status, 0) + 1
    blocking = [check for check in ctx.checks if check.blocks_strict]
    return {
        "schema_version": "1.0",
        "checker": "tools/check_repo_consistency.py",
        "root": str(ctx.root),
        "identities": py_facts.get("identities", {}),
        "summary": {
            "total_checks": len(ctx.checks),
            "status_counts": status_counts,
            "strict_blocking_count": len(blocking),
        },
        "checks": [check.to_json() for check in ctx.checks],
    }


def render_human(report: dict[str, Any]) -> str:
    lines = [
        "omegaprompt repository consistency report",
        f"Root: {report['root']}",
        "",
        "Identities:",
    ]
    identities = report.get("identities", {})
    for key in (
        "github_repo",
        "distribution",
        "primary_import_package",
        "compat_import_package",
        "cli_executables",
        "optional_extras",
        "artifact_schema_version",
        "mcp_tool_names",
    ):
        if key in identities:
            lines.append(f"  {key}: {identities[key]}")

    summary = report["summary"]
    lines.extend(
        [
            "",
            "Summary:",
            f"  total checks: {summary['total_checks']}",
            f"  status counts: {summary['status_counts']}",
            f"  strict blocking findings: {summary['strict_blocking_count']}",
        ]
    )

    findings = [c for c in report["checks"] if c["status"] != "OK"]
    if not findings:
        lines.extend(["", "No drift found."])
        return "\n".join(lines)

    lines.append("")
    lines.append("Findings:")
    severity_order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
    findings.sort(key=lambda c: (severity_order.get(c["severity"], 9), c["id"]))
    for item in findings:
        loc = f" ({item['path']})" if item.get("path") else ""
        lines.append(f"- [{item['severity']}/{item['status']}] {item['id']}{loc}: {item['message']}")
        if item.get("expected") is not None:
            lines.append(f"  expected: {item['expected']}")
        if item.get("actual") is not None:
            lines.append(f"  actual: {item['actual']}")
        if item.get("remediation"):
            lines.append(f"  remediation: {item['remediation']}")
    return "\n".join(lines)


def _write_report(stream: TextIO, text: str) -> None:
    try:
        stream.write(text)
        stream.write("\n")
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        stream.write(safe)
        stream.write("\n")


def main(argv: list[str] | None = None, *, root: Path | str | None = None, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check local omegaprompt repository consistency.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when unapproved drift is found.")
    parser.add_argument("--json-output", type=Path, help="Write machine-readable JSON report to this path.")
    args = parser.parse_args(argv)

    report = run_checks(Path(root) if root is not None else Path.cwd())

    if args.json_output is not None:
        output = args.json_output
        if not output.is_absolute():
            output = (Path(root) if root is not None else Path.cwd()) / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    stream = stdout if stdout is not None else sys.stdout
    _write_report(stream, render_human(report))

    if args.strict and report["summary"]["strict_blocking_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
