#!/usr/bin/env python
"""Post-release verification for omegaprompt.

The final no-network release gate is ``--local-only``. Network checks run only
with ``--network`` and still never publish packages, push tags, or create/edit
GitHub Releases. ``--dry-run`` is informational.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import venv
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TextIO


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import check_repo_consistency, generate_readme_claims, publish_readiness, release_audit, wheel_smoke  # noqa: E402


Status = Literal[
    "OK",
    "WARNING",
    "SKIPPED",
    "NOT_VERIFIED",
    "TOOLING_MISSING",
    "ENVIRONMENT_BLOCKED",
]
FinalStatus = Literal["READY", "VERIFIED", "NOT_READY", "NOT_VERIFIED", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED"]

PROJECT = "omegaprompt"
GITHUB_REPO = "hibou04-ops/omegaprompt"
EXPECTED_IMPORTS = ("omegaprompt", "omegacal")
EXPECTED_CLIS = ("omegaprompt", "omegacal", "omegaprompt-mcp")
EXPECTED_EXTRA = "mcp"


@dataclass
class VerifyCheck:
    id: str
    status: Status
    severity: str
    category: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    remediation: str | None = None
    required: bool = True

    @property
    def blocking(self) -> bool:
        return self.required and self.status in {"NOT_VERIFIED", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED"}

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "details": self.details,
            "remediation": self.remediation,
            "required": self.required,
            "blocking": self.blocking,
        }


class NetworkBlocked(RuntimeError):
    pass


class ResourceMissing(RuntimeError):
    pass


def ok(id: str, message: str, *, category: str, details: dict[str, Any] | None = None) -> VerifyCheck:
    return VerifyCheck(id, "OK", "INFO", category, message, details or {})


def warning(
    id: str,
    message: str,
    *,
    category: str,
    details: dict[str, Any] | None = None,
    remediation: str | None = None,
) -> VerifyCheck:
    return VerifyCheck(id, "WARNING", "WARNING", category, message, details or {}, remediation, required=False)


def skipped(id: str, message: str, *, category: str, details: dict[str, Any] | None = None) -> VerifyCheck:
    return VerifyCheck(id, "SKIPPED", "INFO", category, message, details or {}, required=False)


def not_verified(
    id: str,
    message: str,
    *,
    category: str,
    details: dict[str, Any] | None = None,
    remediation: str | None = None,
) -> VerifyCheck:
    return VerifyCheck(id, "NOT_VERIFIED", "ERROR", category, message, details or {}, remediation)


def tooling_missing(
    id: str,
    message: str,
    *,
    category: str,
    details: dict[str, Any] | None = None,
    remediation: str | None = None,
) -> VerifyCheck:
    return VerifyCheck(
        id,
        "TOOLING_MISSING",
        "ERROR",
        category,
        message,
        details or {},
        remediation or "Install the missing local tooling; this is not verification success.",
    )


def environment_blocked(
    id: str,
    message: str,
    *,
    category: str,
    details: dict[str, Any] | None = None,
    remediation: str | None = None,
) -> VerifyCheck:
    return VerifyCheck(
        id,
        "ENVIRONMENT_BLOCKED",
        "ERROR",
        category,
        message,
        details or {},
        remediation or "Fix network, DNS, auth, or filesystem access; this is not verification success.",
    )


def final_status(checks: list[VerifyCheck], *, network: bool, local_only: bool = False) -> FinalStatus:
    statuses = {check.status for check in checks if check.required}
    if "ENVIRONMENT_BLOCKED" in statuses:
        return "ENVIRONMENT_BLOCKED"
    if "TOOLING_MISSING" in statuses:
        return "TOOLING_MISSING"
    if "NOT_VERIFIED" in statuses:
        return "NOT_READY" if local_only or not network else "NOT_VERIFIED"
    return "VERIFIED" if network else "READY"


def exit_code(status: FinalStatus) -> int:
    if status in {"READY", "VERIFIED"}:
        return 0
    if status in {"TOOLING_MISSING", "ENVIRONMENT_BLOCKED"}:
        return 2
    return 1


def run_verification(
    *,
    root: Path | str = ROOT,
    version: str,
    dry_run: bool = False,
    local_only: bool = False,
    network: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve()
    checks: list[VerifyCheck] = []
    checks.append(_check_version_format(version))
    checks.append(_check_generated_claims(repo_root))
    checks.append(_check_repository_consistency(repo_root))

    if dry_run:
        pass
    elif local_only:
        dist_check = _check_local_dist_artifacts(repo_root, version, required=True)
        checks.append(dist_check)
        if dist_check.status == "OK":
            checks.append(_check_local_wheel_smoke(repo_root, version))
        checks.append(_check_release_audit_local_compat(repo_root))
        checks.append(_check_publish_readiness_local_compat(repo_root))
    elif network:
        pypi_payload: dict[str, Any] | None = None
        pypi_check, pypi_payload = _check_pypi_project_version(version)
        checks.append(pypi_check)
        if pypi_payload is not None:
            checks.append(_check_pypi_distribution_metadata(pypi_payload, version))
            checks.append(_check_pypi_description_claims(pypi_payload))
        else:
            checks.append(not_verified("PYPI_DISTRIBUTION_METADATA", "PyPI distribution metadata was unavailable.", category="pypi"))
            checks.append(not_verified("PYPI_DESCRIPTION_CLAIMS", "PyPI description was unavailable for claim scan.", category="pypi"))
        checks.extend(_check_pypi_install(version))
        checks.append(_check_github_tag(version))
        checks.append(_check_github_release(version))
    else:
        pass

    status = final_status(checks, network=network and not dry_run and not local_only, local_only=local_only)
    counts: dict[str, int] = {}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    return {
        "schema_version": "1.0",
        "tool": "tools/post_release_verify.py",
        "root": str(repo_root),
        "version": version,
        "mode": {
            "dry_run": dry_run,
            "local_only": local_only,
            "network": network,
            "network_checks_included": bool(network and not dry_run and not local_only),
        },
        "final_status": status,
        "summary": {
            "total_checks": len(checks),
            "blocking_checks": sum(1 for check in checks if check.blocking),
            "status_counts": counts,
        },
        "checks": [check.to_json() for check in checks],
        "mutations": {
            "pypi_publish": False,
            "git_tags_created": False,
            "git_tags_pushed": False,
            "github_releases_created_or_edited": False,
        },
    }


def _check_version_format(version: str) -> VerifyCheck:
    if re.fullmatch(r"\d+\.\d+\.\d+(?:[a-zA-Z0-9_.-]+)?", version):
        return ok("VERSION_FORMAT", "Requested version has a valid local version shape.", category="version", details={"version": version})
    return not_verified(
        "VERSION_FORMAT",
        "Requested version is not a valid release-like version string.",
        category="version",
        details={"version": version},
    )


def _check_generated_claims(root: Path) -> VerifyCheck:
    code, messages = generate_readme_claims.generate(root, check=True)
    if code == 0:
        return ok("GENERATED_CLAIMS_FRESH", "Generated README claims are fresh.", category="claims", details={"messages": messages})
    return not_verified(
        "GENERATED_CLAIMS_FRESH",
        "Generated README claims or claim ledger checks failed.",
        category="claims",
        details={"messages": messages},
        remediation="Run `python tools/generate_readme_claims.py` and review public claims.",
    )


def _check_repository_consistency(root: Path) -> VerifyCheck:
    report = check_repo_consistency.run_checks(root)
    if report["summary"]["strict_blocking_count"] == 0:
        return ok("REPOSITORY_CONSISTENCY", "Repository consistency checker reports no strict drift.", category="repo")
    findings = [check for check in report["checks"] if check["status"] != "OK"]
    return not_verified(
        "REPOSITORY_CONSISTENCY",
        "Repository consistency checker found strict drift.",
        category="repo",
        details={"findings": findings},
    )


def _check_local_dist_artifacts(root: Path, version: str, *, required: bool = False) -> VerifyCheck:
    dist = root / "dist"
    expected_wheel = f"{PROJECT}-{version}-py3-none-any.whl"
    expected_sdist = f"{PROJECT}-{version}.tar.gz"
    expected_paths = {
        "wheel": dist / expected_wheel,
        "sdist": dist / expected_sdist,
    }
    if not dist.exists():
        if required:
            return not_verified(
                "LOCAL_DIST_ARTIFACTS",
                "Local dist directory is missing; --local-only requires built wheel and sdist artifacts.",
                category="local",
                details={"expected": [expected_wheel, expected_sdist]},
                remediation="Run `python -m build` before the final local verification gate.",
            )
        return warning(
            "LOCAL_DIST_ARTIFACTS",
                "No local dist directory is present; local-only mode can still verify generated docs.",
                category="local",
                details={"expected": [expected_wheel, expected_sdist]},
            )
    names = sorted(path.name for path in dist.glob("*") if path.is_file())
    missing = [name for name in (expected_wheel, expected_sdist) if name not in names]
    if missing:
        if names or required:
            return not_verified(
                "LOCAL_DIST_ARTIFACTS",
                "Local dist artifacts are missing or do not match the requested version.",
                category="local",
                details={"expected": [expected_wheel, expected_sdist], "actual": names, "missing": missing},
                remediation="Rebuild local artifacts for the requested version before using local artifact verification.",
            )
        return warning(
            "LOCAL_DIST_ARTIFACTS",
            "Local dist artifacts are absent or do not match the requested version.",
            category="local",
                details={"expected": [expected_wheel, expected_sdist], "actual": names, "missing": missing},
        )

    metadata: dict[str, dict[str, str]] = {}
    metadata_notes: list[str] = []
    for kind, path in expected_paths.items():
        try:
            metadata[kind] = _inspect_wheel_metadata(path) if kind == "wheel" else _inspect_sdist_metadata(path)
        except (OSError, zipfile.BadZipFile, tarfile.TarError, ValueError) as exc:
            metadata_notes.append(f"{kind} metadata could not be inspected: {exc}")

    expected = {"name": PROJECT, "version": version}
    mismatches: dict[str, dict[str, str]] = {}
    for key, value in expected.items():
        for kind, parsed in metadata.items():
            actual = parsed.get(key, "")
            expected_value = _normalize_distribution_name(value) if key == "name" else value
            actual_value = _normalize_distribution_name(actual) if key == "name" else actual
            if actual_value != expected_value:
                mismatches[f"{kind}_{key}"] = {"expected": value, "actual": actual}
    if mismatches:
        return not_verified(
            "LOCAL_DIST_ARTIFACTS",
            "Local wheel/sdist metadata does not match the requested release identity.",
            category="local",
            details={"mismatches": mismatches, "metadata": metadata, "metadata_notes": metadata_notes},
        )
    return ok(
        "LOCAL_DIST_ARTIFACTS",
        "Local wheel and sdist filenames match the requested distribution and version.",
        category="local",
        details={
            "artifacts": [expected_wheel, expected_sdist],
            "metadata": metadata,
            "metadata_verified": set(metadata) == {"wheel", "sdist"},
            "metadata_notes": metadata_notes,
        },
    )


def _check_local_wheel_smoke(root: Path, version: str) -> VerifyCheck:
    wheel = root / "dist" / f"{PROJECT}-{version}-py3-none-any.whl"
    if not wheel.exists():
        return not_verified(
            "LOCAL_WHEEL_SMOKE",
            "Local wheel smoke could not run because the expected wheel is missing.",
            category="local",
            details={"wheel": str(wheel)},
        )
    try:
        results = wheel_smoke.run_smoke(wheel=wheel, mode="all")
    except wheel_smoke.SmokeFailure as exc:
        details = {"classification": exc.classification, **exc.details}
        if exc.classification == "TOOLING_MISSING":
            return tooling_missing("LOCAL_WHEEL_SMOKE", str(exc), category="local", details=details)
        if exc.classification == "ENVIRONMENT_BLOCKED":
            return environment_blocked("LOCAL_WHEEL_SMOKE", str(exc), category="local", details=details)
        return not_verified(
            "LOCAL_WHEEL_SMOKE",
            "Local wheel smoke failed.",
            category="local",
            details=details,
            remediation="Run `python tools/wheel_smoke.py --wheel dist/*.whl --mode core` and `--mode mcp` for focused output.",
        )
    return ok(
        "LOCAL_WHEEL_SMOKE",
        "Local wheel smoke passed for core and MCP boundaries.",
        category="local",
        details={"results": [result.__dict__ for result in results]},
    )


def _check_release_audit_local_compat(root: Path) -> VerifyCheck:
    report = release_audit.run_release_audit(root, include_wheel=False)
    mutations = report.get("mutations", {})
    if any(mutations.values()):
        return not_verified(
            "RELEASE_AUDIT_LOCAL_COMPAT",
            "Release audit reported a release-surface mutation.",
            category="release",
            details={"mutations": mutations},
        )
    if report.get("final_status") == "READY":
        return ok(
            "RELEASE_AUDIT_LOCAL_COMPAT",
            "Release audit can run in no-wheel compatibility mode without mutation.",
            category="release",
            details={
                "final_status": report.get("final_status"),
                "blocking_checks": (report.get("summary") or {}).get("blocking_checks"),
            },
        )
    status = str(report.get("final_status"))
    if status in {"TOOLING_MISSING", "ENVIRONMENT_BLOCKED"}:
        factory = tooling_missing if status == "TOOLING_MISSING" else environment_blocked
        return factory(
            "RELEASE_AUDIT_LOCAL_COMPAT",
            "Release audit compatibility check could not complete cleanly.",
            category="release",
            details={"final_status": status, "summary": report.get("summary")},
        )
    return not_verified(
        "RELEASE_AUDIT_LOCAL_COMPAT",
        "Release audit compatibility check is not ready.",
        category="release",
        details={"final_status": status, "summary": report.get("summary")},
    )


def _check_publish_readiness_local_compat(root: Path) -> VerifyCheck:
    report = publish_readiness.build_readiness_report(root, include_wheel=False)
    mutations = report.get("mutations", {})
    if any(mutations.values()):
        return not_verified(
            "PUBLISH_READINESS_LOCAL_COMPAT",
            "Publish readiness reported a release-surface mutation.",
            category="release",
            details={"mutations": mutations},
        )
    if report.get("final_status") == "READY":
        return ok(
            "PUBLISH_READINESS_LOCAL_COMPAT",
            "Publish readiness can run in no-wheel compatibility mode without mutation.",
            category="release",
            details={
                "final_status": report.get("final_status"),
                "blocking_checks": (report.get("summary") or {}).get("blocking_checks"),
            },
        )
    status = str(report.get("final_status"))
    if status in {"TOOLING_MISSING", "ENVIRONMENT_BLOCKED"}:
        factory = tooling_missing if status == "TOOLING_MISSING" else environment_blocked
        return factory(
            "PUBLISH_READINESS_LOCAL_COMPAT",
            "Publish readiness compatibility check could not complete cleanly.",
            category="release",
            details={"final_status": status, "summary": report.get("summary")},
        )
    return not_verified(
        "PUBLISH_READINESS_LOCAL_COMPAT",
        "Publish readiness compatibility check is not ready.",
        category="release",
        details={"final_status": status, "summary": report.get("summary")},
    )


def _inspect_wheel_metadata(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as zf:
        metadata_name = next((name for name in zf.namelist() if name.endswith(".dist-info/METADATA")), None)
        if metadata_name is None:
            raise ValueError("wheel METADATA file not found")
        metadata = zf.read(metadata_name).decode("utf-8", errors="replace")
    return _parse_package_metadata(metadata)


def _inspect_sdist_metadata(path: Path) -> dict[str, str]:
    with tarfile.open(path, "r:gz") as tf:
        pyproject_member = next(
            (
                member
                for member in tf.getmembers()
                if member.name == "pyproject.toml" or member.name.endswith("/pyproject.toml")
            ),
            None,
        )
        if pyproject_member is not None:
            file_obj = tf.extractfile(pyproject_member)
            if file_obj is None:
                raise ValueError("sdist pyproject.toml could not be read")
            data = file_obj.read().decode("utf-8", errors="replace")
            try:
                import tomllib
            except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback only.
                import tomli as tomllib  # type: ignore

            project = tomllib.loads(data).get("project", {})
            return {"name": str(project.get("name", "")), "version": str(project.get("version", ""))}
        pkg_info_member = next(
            (member for member in tf.getmembers() if member.name == "PKG-INFO" or member.name.endswith("/PKG-INFO")),
            None,
        )
        if pkg_info_member is None:
            raise ValueError("sdist pyproject.toml or PKG-INFO not found")
        file_obj = tf.extractfile(pkg_info_member)
        if file_obj is None:
            raise ValueError("sdist PKG-INFO could not be read")
        return _parse_package_metadata(file_obj.read().decode("utf-8", errors="replace"))


def _parse_package_metadata(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized = key.strip().lower()
        if normalized in {"name", "version"} and normalized not in result:
            result[normalized] = value.strip()
    if not result.get("name") or not result.get("version"):
        raise ValueError("package metadata missing Name or Version")
    return result


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _check_pypi_project_version(version: str) -> tuple[VerifyCheck, dict[str, Any] | None]:
    url = f"https://pypi.org/pypi/{PROJECT}/{version}/json"
    try:
        payload = fetch_json(url)
    except ResourceMissing as exc:
        return (
            not_verified(
                "PYPI_PROJECT_VERSION",
                f"PyPI project/version was not found: {PROJECT}=={version}.",
                category="pypi",
                details={"url": url, "exception": str(exc)},
            ),
            None,
        )
    except NetworkBlocked as exc:
        return (
            environment_blocked(
                "PYPI_PROJECT_VERSION",
                "PyPI version check was blocked by network, DNS, or auth restrictions.",
                category="pypi",
                details={"url": url, "exception": str(exc)},
            ),
            None,
        )

    info = payload.get("info") if isinstance(payload, dict) else {}
    name = str(info.get("name", "")) if isinstance(info, dict) else ""
    actual_version = str(info.get("version", "")) if isinstance(info, dict) else ""
    if name.lower() == PROJECT and actual_version == version:
        return (
            ok("PYPI_PROJECT_VERSION", "PyPI project exists for the requested version.", category="pypi", details={"name": name, "version": actual_version}),
            payload,
        )
    return (
        not_verified(
            "PYPI_PROJECT_VERSION",
            "PyPI project metadata does not match the requested release identity.",
            category="pypi",
            details={"expected_name": PROJECT, "expected_version": version, "actual_name": name, "actual_version": actual_version},
        ),
        payload,
    )


def _check_pypi_distribution_metadata(payload: dict[str, Any], version: str) -> VerifyCheck:
    urls = payload.get("urls")
    if not isinstance(urls, list):
        return not_verified("PYPI_DISTRIBUTION_METADATA", "PyPI JSON did not include a urls list.", category="pypi")
    filenames = sorted(str(item.get("filename", "")) for item in urls if isinstance(item, dict))
    expected = {
        f"{PROJECT}-{version}-py3-none-any.whl",
        f"{PROJECT}-{version}.tar.gz",
    }
    missing = sorted(expected - set(filenames))
    if missing:
        return not_verified(
            "PYPI_DISTRIBUTION_METADATA",
            "PyPI wheel/sdist filenames do not match expected release artifacts.",
            category="pypi",
            details={"expected": sorted(expected), "actual": filenames, "missing": missing},
        )
    return ok("PYPI_DISTRIBUTION_METADATA", "PyPI wheel and sdist filenames match expected names.", category="pypi", details={"filenames": filenames})


def _check_pypi_description_claims(payload: dict[str, Any]) -> VerifyCheck:
    info = payload.get("info") if isinstance(payload, dict) else {}
    description = str(info.get("description", "")) if isinstance(info, dict) else ""
    if not description:
        return warning("PYPI_DESCRIPTION_CLAIMS", "PyPI description is empty or unavailable; claim drift could not be scanned.", category="pypi")
    unsupported = _scan_unsupported_description_claims(description)
    if unsupported:
        return not_verified(
            "PYPI_DESCRIPTION_CLAIMS",
            "PyPI description appears to contain unsupported public claim drift.",
            category="pypi",
            details={"findings": unsupported},
            remediation="Update README/PyPI description claims through the claim ledger.",
        )
    return ok("PYPI_DESCRIPTION_CLAIMS", "PyPI description scan found no unsupported claim drift patterns.", category="pypi")


def _check_pypi_install(version: str) -> list[VerifyCheck]:
    try:
        with tempfile.TemporaryDirectory(prefix="omegaprompt-post-release-") as tmp:
            root = Path(tmp)
            core_venv = _create_venv(root / "core")
            mcp_venv = _create_venv(root / "mcp")
            core = _install_and_probe_core(core_venv, version)
            mcp = _install_and_probe_mcp(mcp_venv, version)
            return [core, mcp]
    except MissingTool as exc:
        return [tooling_missing("PYPI_INSTALL", str(exc), category="install", details=exc.details)]
    except EnvironmentBlocked as exc:
        return [environment_blocked("PYPI_INSTALL", str(exc), category="install", details=exc.details)]


class MissingTool(RuntimeError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class EnvironmentBlocked(RuntimeError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


def _create_venv(path: Path) -> Path:
    try:
        venv.EnvBuilder(with_pip=True, clear=True).create(path)
    except Exception as exc:
        raise MissingTool(f"Could not create isolated virtualenv: {exc}", details={"path": str(path)}) from exc
    python = _python_in(path)
    if not python.exists():
        raise MissingTool("Virtualenv Python was not created.", details={"python": str(python)})
    return path


def _install_and_probe_core(venv_dir: Path, version: str) -> VerifyCheck:
    python = _python_in(venv_dir)
    install = _run_subprocess([python, "-m", "pip", "install", f"{PROJECT}=={version}"], cwd=venv_dir)
    if install.returncode != 0:
        return _install_failure_check("PYPI_CORE_INSTALL", "Core install from PyPI failed.", install)
    probe = _run_subprocess([python, "-c", _core_probe_code(version)], cwd=venv_dir)
    if probe.returncode != 0:
        return not_verified("PYPI_CORE_INSTALL", "Core install probe failed.", category="install", details=_proc_details(probe))
    return ok("PYPI_CORE_INSTALL", "Core install from PyPI imports omegaprompt and omegacal and reports the expected version.", category="install", details={"probe": _json_or_text(probe.stdout)})


def _install_and_probe_mcp(venv_dir: Path, version: str) -> VerifyCheck:
    python = _python_in(venv_dir)
    install = _run_subprocess([python, "-m", "pip", "install", f"{PROJECT}[{EXPECTED_EXTRA}]=={version}"], cwd=venv_dir)
    if install.returncode != 0:
        return _install_failure_check("PYPI_MCP_EXTRA_INSTALL", "MCP extra install from PyPI failed.", install)
    probe = _run_subprocess([python, "-c", _mcp_probe_code()], cwd=venv_dir)
    if probe.returncode != 0:
        return not_verified("PYPI_MCP_EXTRA_INSTALL", "MCP extra probe failed.", category="install", details=_proc_details(probe))
    return ok("PYPI_MCP_EXTRA_INSTALL", "MCP extra install exposes the MCP module and omegaprompt-mcp entrypoint.", category="install", details={"probe": _json_or_text(probe.stdout)})


def _install_failure_check(id: str, message: str, proc: subprocess.CompletedProcess[str]) -> VerifyCheck:
    details = _proc_details(proc)
    combined = f"{proc.stdout}\n{proc.stderr}".lower()
    if any(marker in combined for marker in ("temporary failure", "connection", "timed out", "name resolution", "proxy", "ssl")):
        return environment_blocked(id, message, category="install", details=details)
    return not_verified(id, message, category="install", details=details)


def _core_probe_code(version: str) -> str:
    return f"""
import json
import shutil
import omegaprompt
import omegacal
payload = {{
    "omegaprompt_version": omegaprompt.__version__,
    "omegaprompt_import": bool(omegaprompt.__file__),
    "omegacal_import": bool(omegacal.__file__),
    "omegaprompt_cli": bool(shutil.which("omegaprompt")),
    "omegacal_cli": bool(shutil.which("omegacal")),
}}
assert payload["omegaprompt_version"] == {version!r}, payload
assert payload["omegaprompt_import"] and payload["omegacal_import"], payload
assert payload["omegaprompt_cli"] and payload["omegacal_cli"], payload
print(json.dumps(payload, sort_keys=True))
"""


def _mcp_probe_code() -> str:
    return """
import json
import shutil
import omegaprompt.mcp
payload = {
    "mcp_import": bool(omegaprompt.mcp.__file__),
    "omegaprompt_mcp_cli": bool(shutil.which("omegaprompt-mcp")),
}
assert payload["mcp_import"] and payload["omegaprompt_mcp_cli"], payload
print(json.dumps(payload, sort_keys=True))
"""


def _check_github_tag(version: str) -> VerifyCheck:
    tag = f"v{version}"
    url = f"https://api.github.com/repos/{GITHUB_REPO}/git/ref/tags/{tag}"
    try:
        payload = fetch_json(url)
    except ResourceMissing as exc:
        return not_verified("GITHUB_TAG", f"GitHub tag is missing: {tag}.", category="github", details={"url": url, "exception": str(exc)})
    except NetworkBlocked as exc:
        return environment_blocked("GITHUB_TAG", "GitHub tag check was blocked by network, DNS, or auth restrictions.", category="github", details={"url": url, "exception": str(exc)})
    ref = str(payload.get("ref", ""))
    if ref.endswith(f"/{tag}"):
        return ok("GITHUB_TAG", "GitHub tag exists for the requested version.", category="github", details={"tag": tag, "ref": ref})
    return not_verified("GITHUB_TAG", "GitHub tag response did not match the requested version.", category="github", details={"tag": tag, "ref": ref})


def _check_github_release(version: str) -> VerifyCheck:
    tag = f"v{version}"
    url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{tag}"
    try:
        payload = fetch_json(url)
    except ResourceMissing as exc:
        return not_verified(
            "GITHUB_RELEASE",
            f"GitHub Release is explicitly reported missing for tag {tag}.",
            category="github",
            details={"url": url, "tag": tag, "exception": str(exc), "release_missing": True},
            remediation="Create the GitHub Release out of band if release policy requires it; this verifier will not create it.",
        )
    except NetworkBlocked as exc:
        return environment_blocked("GITHUB_RELEASE", "GitHub Release check was blocked by network, DNS, or auth restrictions.", category="github", details={"url": url, "exception": str(exc)})
    if str(payload.get("tag_name", "")) == tag:
        return ok("GITHUB_RELEASE", "GitHub Release exists for the requested tag.", category="github", details={"tag": tag, "html_url": payload.get("html_url")})
    return not_verified("GITHUB_RELEASE", "GitHub Release response did not match the requested tag.", category="github", details={"tag": tag, "actual": payload.get("tag_name")})


def fetch_json(url: str, *, timeout: int = 20) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "omegaprompt-post-release-verify"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - explicit opt-in network mode.
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ResourceMissing(f"HTTP 404 for {url}") from exc
        if exc.code in {401, 403, 429}:
            raise NetworkBlocked(f"HTTP {exc.code} for {url}") from exc
        raise NetworkBlocked(f"HTTP {exc.code} for {url}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise NetworkBlocked(str(exc)) from exc


def _run_subprocess(cmd: list[str | os.PathLike[str]], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [str(part) for part in cmd],
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MissingTool(f"Command not found: {cmd[0]}", details={"exception": str(exc)}) from exc


def _python_in(venv_dir: Path) -> Path:
    scripts = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    return venv_dir / scripts / exe


def _proc_details(proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def _json_or_text(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _scan_unsupported_description_claims(description: str) -> list[str]:
    findings: list[str] = []
    for line_no, line in enumerate(description.splitlines(), start=1):
        lowered = line.lower()
        if re.search(r"\b\d[\d,]*(?:\+)?\s+(downloads?|users?|adopters?|installs?)\b", line, flags=re.IGNORECASE):
            findings.append(f"line {line_no}: unsupported adoption/download claim")
        for phrase in ("best provider", "best model", "provider is superior", "model is superior", "prompt is superior"):
            if phrase in lowered:
                findings.append(f"line {line_no}: unsupported superiority claim pattern {phrase!r}")
    return findings


def render_human(report: dict[str, Any]) -> str:
    lines = [
        "omegaprompt post-release verification report",
        f"Root: {report['root']}",
        f"Version: {report['version']}",
        f"Mode: dry_run={report['mode']['dry_run']} local_only={report['mode']['local_only']} network={report['mode']['network']}",
        f"Network checks included: {report['mode']['network_checks_included']}",
        f"Final status: {report['final_status']}",
        "",
        "Summary:",
        f"  total checks: {report['summary']['total_checks']}",
        f"  blocking checks: {report['summary']['blocking_checks']}",
        f"  status counts: {report['summary']['status_counts']}",
        "",
        "Checks:",
    ]
    for check in report["checks"]:
        lines.append(f"- [{check['status']}] {check['id']}: {check['message']}")
        if check.get("remediation"):
            lines.append(f"  remediation: {check['remediation']}")
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
    parser = argparse.ArgumentParser(description="Verify an omegaprompt release after publication without mutating release surfaces.")
    parser.add_argument("--version", required=True, help="Version to verify, for example 1.7.4.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Informational no-network run; does not require local dist artifacts and is not the final local gate.")
    mode.add_argument("--local-only", action="store_true", help="Final no-network local gate; requires local dist artifacts and reports no skipped network check.")
    mode.add_argument("--network", action="store_true", help="Enable PyPI/GitHub and PyPI install verification.")
    parser.add_argument("--json-output", type=Path, help="Write machine-readable JSON report to this path.")
    args = parser.parse_args(argv)

    repo_root = Path(root).resolve() if root is not None else Path.cwd().resolve()
    report = run_verification(
        root=repo_root,
        version=args.version,
        dry_run=args.dry_run,
        local_only=args.local_only,
        network=args.network,
    )
    if args.json_output:
        output = args.json_output if args.json_output.is_absolute() else repo_root / args.json_output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _write_text(stdout or sys.stdout, render_human(report))
    return exit_code(report["final_status"])


if __name__ == "__main__":
    raise SystemExit(main())
