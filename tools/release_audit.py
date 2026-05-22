#!/usr/bin/env python
"""Offline release audit for omegaprompt.

The audit is intentionally read-only with respect to release surfaces: it does
not publish packages, create tags, push tags, or create GitHub Releases. Wheel
build/smoke checks use local temporary directories and local wheels only.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TextIO

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback only.
    import tomli as tomllib  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import check_repo_consistency, generate_readme_claims, reproduce_golden_reference, wheel_smoke  # noqa: E402


CheckStatus = Literal["OK", "WARNING", "NOT_READY", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED"]
FinalStatus = Literal["READY", "NOT_READY", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED"]


@dataclass
class AuditCheck:
    id: str
    status: CheckStatus
    severity: str
    category: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    remediation: str | None = None

    @property
    def blocking(self) -> bool:
        return self.status in {"NOT_READY", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED"}

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "details": self.details,
            "remediation": self.remediation,
            "blocking": self.blocking,
        }


def ok(id: str, message: str, *, category: str, details: dict[str, Any] | None = None) -> AuditCheck:
    return AuditCheck(id, "OK", "INFO", category, message, details or {})


def warning(
    id: str,
    message: str,
    *,
    category: str,
    details: dict[str, Any] | None = None,
    remediation: str | None = None,
) -> AuditCheck:
    return AuditCheck(id, "WARNING", "WARNING", category, message, details or {}, remediation)


def not_ready(
    id: str,
    message: str,
    *,
    category: str,
    details: dict[str, Any] | None = None,
    remediation: str | None = None,
) -> AuditCheck:
    return AuditCheck(id, "NOT_READY", "ERROR", category, message, details or {}, remediation)


def tooling_missing(
    id: str,
    message: str,
    *,
    category: str,
    details: dict[str, Any] | None = None,
    remediation: str | None = None,
) -> AuditCheck:
    return AuditCheck(
        id,
        "TOOLING_MISSING",
        "ERROR",
        category,
        message,
        details or {},
        remediation or "Install the missing local tooling; this is not release approval.",
    )


def environment_blocked(
    id: str,
    message: str,
    *,
    category: str,
    details: dict[str, Any] | None = None,
    remediation: str | None = None,
) -> AuditCheck:
    return AuditCheck(
        id,
        "ENVIRONMENT_BLOCKED",
        "ERROR",
        category,
        message,
        details or {},
        remediation or "Fix the local environment or filesystem access; this is not release approval.",
    )


def final_status(checks: list[AuditCheck]) -> FinalStatus:
    statuses = {check.status for check in checks}
    if "ENVIRONMENT_BLOCKED" in statuses:
        return "ENVIRONMENT_BLOCKED"
    if "TOOLING_MISSING" in statuses:
        return "TOOLING_MISSING"
    if "NOT_READY" in statuses:
        return "NOT_READY"
    return "READY"


def strict_exit_code(status: FinalStatus) -> int:
    if status == "READY":
        return 0
    if status in {"TOOLING_MISSING", "ENVIRONMENT_BLOCKED"}:
        return 2
    return 1


def run_release_audit(root: Path | str = ROOT, *, include_wheel: bool = True) -> dict[str, Any]:
    repo_root = Path(root).resolve()
    checks: list[AuditCheck] = []

    facts = _version_facts(repo_root)
    version = str(facts.get("pyproject_version") or "")
    checks.append(_check_version_alignment(facts))
    checks.append(_check_branch_cleanliness(repo_root))
    checks.append(_check_claim_ledger(repo_root))
    checks.append(_check_generated_claims(repo_root))
    checks.append(_check_reference_artifacts(repo_root))
    checks.append(_check_artifact_integrity(repo_root))
    checks.append(_check_provider_docs_code(repo_root))
    checks.append(_check_readme_badges(repo_root))
    checks.append(_check_no_default_live_tests(repo_root))
    checks.append(_check_repository_consistency(repo_root))
    checks.append(_check_git_tag_release_state(repo_root, version))

    if include_wheel:
        with tempfile.TemporaryDirectory(prefix="omegaprompt-release-audit-wheel-") as tmp:
            build_check, wheel = _check_wheel_build(repo_root, Path(tmp))
            checks.append(build_check)
            if wheel is not None:
                checks.append(_check_wheel_smoke(wheel, "core"))
                checks.append(_check_wheel_smoke(wheel, "mcp"))
            else:
                checks.append(_wheel_smoke_not_run("core", build_check.status))
                checks.append(_wheel_smoke_not_run("mcp", build_check.status))

    status = final_status(checks)
    status_counts: dict[str, int] = {}
    for check in checks:
        status_counts[check.status] = status_counts.get(check.status, 0) + 1
    return {
        "schema_version": "1.0",
        "tool": "tools/release_audit.py",
        "root": str(repo_root),
        "final_status": status,
        "version": facts,
        "summary": {
            "total_checks": len(checks),
            "blocking_checks": sum(1 for check in checks if check.blocking),
            "status_counts": status_counts,
        },
        "checks": [check.to_json() for check in checks],
        "deferred_external_checks": _deferred_external_checks(version),
        "mutations": {
            "pypi_publish": False,
            "git_tags_created": False,
            "git_tags_pushed": False,
            "github_releases_created_or_edited": False,
        },
    }


def _version_facts(root: Path) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    try:
        pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        facts["pyproject_version"] = pyproject.get("project", {}).get("version")
    except Exception as exc:
        facts["pyproject_error"] = str(exc)

    try:
        init_text = (root / "src" / "omegaprompt" / "__init__.py").read_text(encoding="utf-8")
        match = re.search(r'^__version__\s*=\s*"([^"]+)"', init_text, flags=re.MULTILINE)
        facts["package_version"] = match.group(1) if match else None
    except Exception as exc:
        facts["package_error"] = str(exc)

    try:
        readme = (root / "README.md").read_text(encoding="utf-8")
        badge_match = re.search(r"pypi-([0-9]+\.[0-9]+\.[0-9]+)-blue\.svg", readme)
        facts["readme_badge_version"] = badge_match.group(1) if badge_match else None
    except Exception as exc:
        facts["readme_error"] = str(exc)

    try:
        changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
        change_match = re.search(r"^## \[([0-9]+\.[0-9]+\.[0-9]+)\]", changelog, flags=re.MULTILINE)
        facts["changelog_latest_version"] = change_match.group(1) if change_match else None
    except Exception as exc:
        facts["changelog_error"] = str(exc)
    return facts


def _check_version_alignment(facts: dict[str, Any]) -> AuditCheck:
    required = {
        "pyproject_version": facts.get("pyproject_version"),
        "package_version": facts.get("package_version"),
        "readme_badge_version": facts.get("readme_badge_version"),
        "changelog_latest_version": facts.get("changelog_latest_version"),
    }
    if any(value is None for value in required.values()):
        return environment_blocked(
            "VERSION_ALIGNMENT",
            "Could not read all local version sources.",
            category="version",
            details=facts,
        )
    versions = set(required.values())
    if len(versions) == 1:
        return ok("VERSION_ALIGNMENT", "pyproject, __version__, README badge, and CHANGELOG align.", category="version", details=required)
    return not_ready(
        "VERSION_ALIGNMENT",
        "Version sources are not aligned.",
        category="version",
        details=required,
        remediation="Align pyproject.toml, src/omegaprompt/__init__.py, README.md PyPI badge, and CHANGELOG.md.",
    )


def _check_branch_cleanliness(root: Path) -> AuditCheck:
    proc = _git(root, ["status", "--porcelain", "--branch"])
    if proc is None:
        return tooling_missing("GIT_BRANCH_STATE", "git is not available for branch cleanliness checks.", category="git")
    if proc.returncode != 0:
        return warning(
            "GIT_BRANCH_STATE",
            "Branch cleanliness could not be checked.",
            category="git",
            details={"stderr": proc.stderr, "returncode": proc.returncode},
            remediation="Rerun from a readable git checkout before release.",
        )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    dirty = [line for line in lines if not line.startswith("##")]
    if dirty:
        return warning(
            "GIT_BRANCH_STATE",
            "Working tree has uncommitted changes; release audit reports this but does not mutate it.",
            category="git",
            details={"status": lines},
            remediation="Commit or intentionally account for these changes before publishing.",
        )
    return ok("GIT_BRANCH_STATE", "Working tree is clean.", category="git", details={"status": lines})


def _check_claim_ledger(root: Path) -> AuditCheck:
    try:
        ledger = generate_readme_claims.load_ledger(root)
    except ValueError as exc:
        return not_ready("CLAIM_LEDGER_VALID", str(exc), category="claims")
    errors = generate_readme_claims.validate_ledger(ledger, root)
    errors.extend(generate_readme_claims.scan_unsupported_public_claims(ledger, root))
    if errors:
        return not_ready(
            "CLAIM_LEDGER_VALID",
            "Claim ledger validation or public-claim scan failed.",
            category="claims",
            details={"errors": errors},
            remediation="Update docs/claims/public_claim_ledger.json or remove unsupported public claims.",
        )
    return ok("CLAIM_LEDGER_VALID", "Claim ledger is valid and public docs have no unsupported exact claims.", category="claims")


def _check_generated_claims(root: Path) -> AuditCheck:
    code, messages = generate_readme_claims.generate(root, check=True)
    if code == 0:
        return ok("README_CLAIMS_FRESH", "Generated README claim document is fresh.", category="claims", details={"messages": messages})
    return not_ready(
        "README_CLAIMS_FRESH",
        "Generated README claim document is stale or invalid.",
        category="claims",
        details={"messages": messages},
        remediation="Run `python tools/generate_readme_claims.py` and review the generated diff.",
    )


def _check_reference_artifacts(root: Path) -> AuditCheck:
    try:
        harness = reproduce_golden_reference._load_harness()
        errors = reproduce_golden_reference._check(harness)
    except ImportError as exc:
        return tooling_missing("REFERENCE_ARTIFACTS_FRESH", f"Could not load golden reference harness: {exc}", category="artifact")
    except Exception as exc:
        return not_ready("REFERENCE_ARTIFACTS_FRESH", f"Golden reference check failed: {exc}", category="artifact")
    if not errors:
        return ok("REFERENCE_ARTIFACTS_FRESH", "Golden reference artifacts are fresh.", category="artifact")
    if any(error.startswith("ENVIRONMENT_BLOCKED") for error in errors):
        return environment_blocked("REFERENCE_ARTIFACTS_FRESH", "Golden reference artifacts are inaccessible.", category="artifact", details={"errors": errors})
    return not_ready(
        "REFERENCE_ARTIFACTS_FRESH",
        "Golden reference artifacts are stale.",
        category="artifact",
        details={"errors": errors},
        remediation="Run `python tools/reproduce_golden_reference.py` and review deterministic artifact changes.",
    )


def _check_artifact_integrity(root: Path) -> AuditCheck:
    try:
        from omegaprompt.core.artifact_integrity import check_artifact_integrity
    except ModuleNotFoundError as exc:
        return tooling_missing("REFERENCE_ARTIFACT_INTEGRITY", f"Artifact integrity module is unavailable: {exc}", category="artifact")

    artifact = root / "examples" / "reference" / "reference_artifact.json"
    report = check_artifact_integrity(artifact)
    details = json.loads(report.model_dump_json())
    if report.environment_blocked:
        return environment_blocked("REFERENCE_ARTIFACT_INTEGRITY", "Reference artifact is inaccessible.", category="artifact", details=details)
    if report.valid and report.strict_blocking_findings == 0:
        return ok("REFERENCE_ARTIFACT_INTEGRITY", "Reference artifact integrity check passes.", category="artifact", details={"release_approved": report.release_approved, "canonical_sha256": report.canonical_sha256})
    return not_ready(
        "REFERENCE_ARTIFACT_INTEGRITY",
        "Reference artifact integrity check failed.",
        category="artifact",
        details=details,
        remediation="Fix the artifact through the existing CalibrationArtifact validation path; do not weaken validation.",
    )


def _check_provider_docs_code(root: Path) -> AuditCheck:
    try:
        from unittest.mock import MagicMock

        from omegaprompt.providers import make_provider
        from omegaprompt.providers.base import provider_capabilities
    except ModuleNotFoundError as exc:
        return tooling_missing("PROVIDER_DOCS_CODE_CONSISTENCY", f"Provider modules are unavailable: {exc}", category="providers")

    doc = (root / "docs" / "provider-capabilities.md").read_text(encoding="utf-8")
    gemini = provider_capabilities(make_provider("gemini", client=MagicMock()))
    local = provider_capabilities(make_provider("local", client=MagicMock(), base_url="http://localhost:11434/v1"))
    failures: list[str] = []
    if gemini.placeholder:
        failures.append("Gemini provider reports placeholder=True.")
    if gemini.ship_grade_judge:
        failures.append("Gemini provider reports ship_grade_judge=True without a doc contract change.")
    if "| Gemini | Tier 2 cloud-grade | Implemented target adapter | Not ship-grade judge |" not in doc:
        failures.append("Provider docs do not describe Gemini as implemented target / non-ship-grade judge.")
    if not local.experimental or local.ship_grade_judge:
        failures.append("Local provider capabilities no longer report experimental non-ship-grade judge behavior.")
    if "Experimental target path | Exploration-grade judge only" not in doc:
        failures.append("Provider docs do not describe local adapters as exploration-grade.")
    if failures:
        return not_ready(
            "PROVIDER_DOCS_CODE_CONSISTENCY",
            "Provider docs drifted from provider capability code.",
            category="providers",
            details={"failures": failures},
            remediation="Update docs/provider-capabilities.md and provider contract tests together.",
        )
    return ok("PROVIDER_DOCS_CODE_CONSISTENCY", "Provider docs match capability code for Gemini and local adapters.", category="providers")


def _check_readme_badges(root: Path) -> AuditCheck:
    readme = (root / "README.md").read_text(encoding="utf-8")
    badge_lines = [line.strip() for line in readme.splitlines()[:25] if line.strip().startswith("[![")]
    expected = check_repo_consistency.README_BADGES
    labels_ok = len(badge_lines) == len(expected)
    tokens_ok = labels_ok and all(label in line and token in line for (label, token), line in zip(expected, badge_lines))
    if labels_ok and tokens_ok:
        return ok("README_BADGE_COMPOSITION", "README.md top badge composition is unchanged.", category="docs", details={"badge_count": len(badge_lines)})
    return not_ready(
        "README_BADGE_COMPOSITION",
        "README.md top badge composition changed.",
        category="docs",
        details={"actual": badge_lines, "expected": expected},
        remediation="Preserve the exact CI/license/python/PyPI/tests/schema/MCP/framework badge row.",
    )


def _check_no_default_live_tests(root: Path) -> AuditCheck:
    workflow_path = root / ".github" / "workflows" / "ci.yml"
    try:
        workflow = workflow_path.read_text(encoding="utf-8")
    except OSError as exc:
        return environment_blocked("DEFAULT_CI_NO_LIVE_TESTS", f"Cannot read CI workflow: {exc}", category="ci")

    required = [
        'OMEGAPROMPT_LIVE_PROVIDER_TESTS: "0"',
        'python -m pytest -q -m "not live"',
    ]
    missing = [item for item in required if item not in workflow]
    live_files = sorted((root / "tests" / "live").glob("test_*.py"))
    unguarded_live = [
        path.relative_to(root).as_posix()
        for path in live_files
        if "OMEGAPROMPT_LIVE_PROVIDER_TESTS" not in path.read_text(encoding="utf-8")
    ]
    if missing or unguarded_live:
        return not_ready(
            "DEFAULT_CI_NO_LIVE_TESTS",
            "Default CI or live tests no longer enforce explicit live-provider opt-in.",
            category="ci",
            details={"missing_workflow_fragments": missing, "unguarded_live_tests": unguarded_live},
            remediation="Keep live provider tests behind OMEGAPROMPT_LIVE_PROVIDER_TESTS=1 and excluded from default CI.",
        )
    return ok("DEFAULT_CI_NO_LIVE_TESTS", "Default CI excludes live provider tests and live tests require explicit opt-in.", category="ci")


def _check_repository_consistency(root: Path) -> AuditCheck:
    report = check_repo_consistency.run_checks(root)
    blocking = report["summary"]["strict_blocking_count"]
    if blocking == 0:
        return ok("REPOSITORY_CONSISTENCY", "Repository consistency checker reports no strict drift.", category="repo", details={"total_checks": report["summary"]["total_checks"]})
    findings = [check for check in report["checks"] if check["status"] != "OK"]
    return not_ready(
        "REPOSITORY_CONSISTENCY",
        "Repository consistency checker found strict drift.",
        category="repo",
        details={"blocking": blocking, "findings": findings},
        remediation="Run `python tools/check_repo_consistency.py --strict` and fix reported drift.",
    )


def _check_git_tag_release_state(root: Path, version: str) -> AuditCheck:
    if not version:
        return warning("GIT_TAG_RELEASE_STATE", "Version is unknown; tag/release state could not be interpreted.", category="release")
    tag = f"v{version}"
    proc = _git(root, ["tag", "--list", tag])
    if proc is None:
        return tooling_missing("GIT_TAG_RELEASE_STATE", "git is not available for tag state checks.", category="release")
    if proc.returncode != 0:
        return warning(
            "GIT_TAG_RELEASE_STATE",
            "Local git tag state could not be checked.",
            category="release",
            details={"stderr": proc.stderr, "returncode": proc.returncode},
        )
    tag_exists = tag in {line.strip() for line in proc.stdout.splitlines()}
    marker_candidates = [
        root / ".github" / "releases" / f"{tag}.md",
        root / "docs" / "release" / "releases" / f"{tag}.md",
        root / "docs" / "release" / f"{tag}.md",
    ]
    release_marker = next((path for path in marker_candidates if path.exists()), None)
    details = {
        "tag": tag,
        "local_tag_exists": tag_exists,
        "release_marker": str(release_marker.relative_to(root)) if release_marker else None,
        "release_marker_candidates": [str(path.relative_to(root)) for path in marker_candidates],
        "network_or_github_mutation": False,
    }
    if tag_exists and release_marker is None:
        return ok(
            "GIT_TAG_RELEASE_STATE",
            "Local tag exists; GitHub Release existence is deferred to post-release network verification.",
            category="release",
            details=details,
        )
    if tag_exists:
        return ok(
            "GIT_TAG_RELEASE_STATE",
            "Local tag exists and a local release marker was found; GitHub Release network existence remains deferred.",
            category="release",
            details=details,
        )
    return ok("GIT_TAG_RELEASE_STATE", "No local tag exists for the current version.", category="release", details=details)


def _deferred_external_checks(version: str) -> list[dict[str, Any]]:
    release_version = version or "<version>"
    return [
        {
            "id": "GITHUB_RELEASE_NETWORK_VERIFICATION",
            "status": "DEFERRED",
            "category": "github",
            "message": "GitHub Release existence is not verified by local release_audit.",
            "command": (
                "python tools/post_release_verify.py "
                f"--version {release_version} --network --json-output build/post_release_verify_network.json"
            ),
            "required_after_release": True,
            "network_required": True,
            "mutates_release_surfaces": False,
        }
    ]


def _check_wheel_build(root: Path, outdir: Path) -> tuple[AuditCheck, Path | None]:
    try:
        import build  # noqa: F401
    except ModuleNotFoundError as exc:
        return (
            tooling_missing(
                "WHEEL_BUILD",
                "The 'build' package is required for wheel build checks.",
                category="wheel",
                details={"exception": str(exc)},
                remediation="Install with `python -m pip install build`.",
            ),
            None,
        )
    proc = _run([sys.executable, "-m", "build", "--wheel", "--outdir", str(outdir)], cwd=root)
    if proc is None:
        return (tooling_missing("WHEEL_BUILD", "Python executable for wheel build was not found.", category="wheel"), None)
    if proc.returncode != 0:
        lowered = f"{proc.stdout}\n{proc.stderr}".lower()
        if _looks_like_build_tooling_failure(lowered):
            check = tooling_missing(
                "WHEEL_BUILD",
                "Wheel build tooling or build backend dependency is unavailable in this environment.",
                category="wheel",
                details=_proc_details(proc),
                remediation="Install local build tooling, including `build` and the pyproject build backend, then rerun; this is not release approval.",
            )
        else:
            check = not_ready("WHEEL_BUILD", "Wheel build failed.", category="wheel", details=_proc_details(proc))
        return (check, None)
    wheels = sorted(outdir.glob("*.whl"), key=lambda path: path.stat().st_mtime)
    if not wheels:
        return (not_ready("WHEEL_BUILD", "Wheel build completed but produced no .whl file.", category="wheel", details=_proc_details(proc)), None)
    return (ok("WHEEL_BUILD", "Wheel build succeeded.", category="wheel", details={"wheel": str(wheels[-1])}), wheels[-1])


def _looks_like_build_tooling_failure(text: str) -> bool:
    markers = (
        "no module named build",
        "backend 'hatchling.build' is not available",
        "cannot import 'hatchling.build'",
        "installing packages in isolated environment",
        "build-system.requires",
        "build backend",
        "hatchling",
    )
    return any(marker in text for marker in markers)


def _check_wheel_smoke(wheel: Path, mode: Literal["core", "mcp"]) -> AuditCheck:
    check_id = f"WHEEL_SMOKE_{mode.upper()}"
    try:
        results = wheel_smoke.run_smoke(wheel=wheel, mode=mode)
    except wheel_smoke.SmokeFailure as exc:
        details = {"classification": exc.classification, **exc.details}
        if exc.classification == "TOOLING_MISSING":
            return tooling_missing(check_id, str(exc), category="wheel", details=details)
        if exc.classification == "ENVIRONMENT_BLOCKED":
            return environment_blocked(check_id, str(exc), category="wheel", details=details)
        return not_ready(
            check_id,
            f"{mode} wheel smoke failed.",
            category="wheel",
            details=details,
            remediation="Fix packaging or optional-extra boundaries before publishing.",
        )
    return ok(check_id, f"{mode} wheel smoke passed.", category="wheel", details={"results": [result.__dict__ for result in results]})


def _wheel_smoke_not_run(mode: Literal["core", "mcp"], build_status: CheckStatus) -> AuditCheck:
    check_id = f"WHEEL_SMOKE_{mode.upper()}"
    message = f"{mode} wheel smoke was not run because no wheel was available."
    if build_status == "TOOLING_MISSING":
        return tooling_missing(check_id, message, category="wheel")
    if build_status == "ENVIRONMENT_BLOCKED":
        return environment_blocked(check_id, message, category="wheel")
    return not_ready(check_id, message, category="wheel")


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return None


def _git(root: Path, args: list[str]) -> subprocess.CompletedProcess[str] | None:
    return _run(["git", "-c", f"safe.directory={root.as_posix()}", *args], cwd=root)


def _proc_details(proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def render_human(report: dict[str, Any]) -> str:
    lines = [
        "omegaprompt release audit report",
        f"Root: {report['root']}",
        f"Final status: {report['final_status']}",
        "",
        "Version:",
    ]
    for key, value in report.get("version", {}).items():
        lines.append(f"  {key}: {value}")
    lines.extend(
        [
            "",
            "Summary:",
            f"  total checks: {report['summary']['total_checks']}",
            f"  blocking checks: {report['summary']['blocking_checks']}",
            f"  status counts: {report['summary']['status_counts']}",
            "",
            "Checks:",
        ]
    )
    for check in report["checks"]:
        lines.append(f"- [{check['status']}] {check['id']}: {check['message']}")
        if check.get("remediation"):
            lines.append(f"  remediation: {check['remediation']}")
    deferred = report.get("deferred_external_checks", [])
    if deferred:
        lines.extend(["", "Deferred external checks:"])
        for item in deferred:
            lines.append(f"- [{item['status']}] {item['id']}: {item['message']}")
            if item.get("command"):
                lines.append(f"  command: {item['command']}")
    lines.extend(
        [
            "",
            "Release mutations:",
            "  PyPI publish: false",
            "  git tags created/pushed: false",
            "  GitHub Releases created/edited: false",
        ]
    )
    return "\n".join(lines)


def _write_text(stream: TextIO, text: str) -> None:
    try:
        stream.write(text + "\n")
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        stream.write(text.encode(encoding, errors="replace").decode(encoding, errors="replace") + "\n")


def main(argv: list[str] | None = None, *, root: Path | str | None = None, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local omegaprompt release audit without publishing.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless the audit is READY.")
    parser.add_argument("--json-output", type=Path, help="Write the machine-readable audit report to this path.")
    args = parser.parse_args(argv)

    repo_root = Path(root).resolve() if root is not None else Path.cwd().resolve()
    report = run_release_audit(repo_root)
    if args.json_output:
        output = args.json_output if args.json_output.is_absolute() else repo_root / args.json_output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _write_text(stdout or sys.stdout, render_human(report))
    if args.strict:
        return strict_exit_code(report["final_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
