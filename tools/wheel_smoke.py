"""Offline wheel smoke checks for core and MCP package boundaries.

The smoke installs a locally built wheel into temporary virtualenvs using
``--no-index --no-deps``. Dependencies are expected to be available from the
current CI environment through ``--system-site-packages``; missing dependency or
venv/build tooling is reported as ``TOOLING_MISSING`` rather than release
approval.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import site
import subprocess
import sys
import tempfile
import venv
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback only
    import tomli as tomllib  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ARTIFACT = ROOT / "examples" / "reference" / "reference_artifact.json"
EXPECTED_MCP_TOOLS = {
    "calibrate",
    "evaluate",
    "report",
    "diff",
    "measure_sensitivity",
    "grade",
    "preflight",
    "classify_traps",
}


Classification = Literal["OK", "TOOLING_MISSING", "ENVIRONMENT_BLOCKED", "SMOKE_FAILED"]


@dataclass
class SmokeResult:
    name: str
    classification: Classification
    message: str
    details: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.classification == "OK"


class SmokeFailure(RuntimeError):
    def __init__(
        self,
        classification: Classification,
        message: str,
        *,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.classification = classification
        self.details = details or {}


def _python_in(venv_dir: Path) -> Path:
    scripts = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    return venv_dir / scripts / exe


def _script_in(venv_dir: Path, name: str) -> Path:
    scripts = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    return venv_dir / scripts / f"{name}{suffix}"


def _run(
    cmd: list[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    classification: Classification = "SMOKE_FAILED",
) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            [str(part) for part in cmd],
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            env=env,
        )
    except FileNotFoundError as exc:
        raise SmokeFailure(
            "TOOLING_MISSING",
            f"Command not found: {cmd[0]}",
            details={"exception": str(exc)},
        ) from exc
    if proc.returncode != 0:
        raise SmokeFailure(
            classification,
            f"Command failed ({proc.returncode}): {' '.join(str(p) for p in cmd)}",
            details={
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
            },
        )
    return proc


def _build_wheel() -> Path:
    try:
        import build  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SmokeFailure(
            "TOOLING_MISSING",
            "The 'build' package is required to build a wheel. Install with `python -m pip install build`.",
        ) from exc

    dist_dir = ROOT / "dist"
    before = {p.resolve() for p in dist_dir.glob("*.whl")} if dist_dir.exists() else set()
    _run(
        [sys.executable, "-m", "build", "--wheel"],
        cwd=ROOT,
        classification="TOOLING_MISSING",
    )
    after = {p.resolve() for p in dist_dir.glob("*.whl")}
    new_wheels = sorted(after - before, key=lambda p: p.stat().st_mtime)
    candidates = new_wheels or sorted(after, key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SmokeFailure("TOOLING_MISSING", "Wheel build completed but no dist/*.whl was found.")
    return candidates[-1]


def _expand_wheels(patterns: list[str]) -> list[Path]:
    wheels: list[Path] = []
    for pattern in patterns:
        expanded = [Path(p) for p in glob.glob(pattern)]
        if expanded:
            wheels.extend(expanded)
        else:
            wheels.append(Path(pattern))
    unique = sorted({p.resolve() for p in wheels})
    return [p for p in unique if p.exists() and p.suffix == ".whl"]


def _resolve_wheel(patterns: list[str] | None) -> Path:
    if not patterns:
        return _build_wheel()
    wheels = _expand_wheels(patterns)
    if not wheels:
        raise SmokeFailure(
            "ENVIRONMENT_BLOCKED",
            f"No wheel matched --wheel pattern(s): {patterns}",
        )
    if len(wheels) > 1:
        raise SmokeFailure(
            "ENVIRONMENT_BLOCKED",
            "Multiple wheels matched --wheel; pass exactly one wheel path.",
            details={"wheels": [str(p) for p in wheels]},
        )
    return wheels[0]


def _create_venv(root: Path, name: str) -> Path:
    venv_dir = root / name
    try:
        builder = venv.EnvBuilder(with_pip=True, system_site_packages=False, clear=True)
        builder.create(venv_dir)
    except Exception as exc:
        raise SmokeFailure(
            "TOOLING_MISSING",
            f"Could not create virtualenv: {exc}",
        ) from exc
    python = _python_in(venv_dir)
    if not python.exists():
        raise SmokeFailure("TOOLING_MISSING", f"Virtualenv python not found at {python}")
    _link_current_dependency_paths(venv_dir)
    return venv_dir


def _current_dependency_paths() -> list[Path]:
    candidates: list[str] = []
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass
    candidates.extend(entry for entry in sys.path if entry)
    paths: list[Path] = []
    for candidate in candidates:
        path = Path(candidate).resolve()
        if not path.exists():
            continue
        if path == ROOT:
            # Do not inherit the repository checkout itself. The wheel install
            # must provide omegaprompt/omegacal. A local .venv site-packages
            # directory is allowed so dependency wheels already installed for
            # CI can be reused without network.
            continue
        if path not in paths:
            paths.append(path)
    return paths


def _venv_purelib(venv_dir: Path) -> Path:
    proc = _run(
        [
            _python_in(venv_dir),
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ],
        classification="TOOLING_MISSING",
    )
    return Path(proc.stdout.strip())


def _link_current_dependency_paths(venv_dir: Path) -> None:
    dependency_paths = _current_dependency_paths()
    if not dependency_paths:
        return
    purelib = _venv_purelib(venv_dir)
    purelib.mkdir(parents=True, exist_ok=True)
    pth = purelib / "_omegaprompt_wheel_smoke_deps.pth"
    pth.write_text(
        "".join(f"{path}\n" for path in dependency_paths),
        encoding="utf-8",
    )


def _install_wheel(venv_dir: Path, wheel: Path, *, extra: str | None = None) -> None:
    python = _python_in(venv_dir)
    spec = str(wheel)
    if extra:
        spec = f"{spec}[{extra}]"
    _run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            spec,
        ],
        classification="TOOLING_MISSING",
    )


def _blocked_mcp_import_code() -> str:
    return r"""
import importlib.abc
import json
import pathlib
import sys

class BlockMcp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mcp" or fullname.startswith("mcp."):
            raise ModuleNotFoundError("No module named 'mcp'", name="mcp")
        return None

sys.meta_path.insert(0, BlockMcp())
import omegaprompt
import omegacal
from omegaprompt.cli import app as primary_cli
from omegacal.cli import app as alias_cli
assert primary_cli is not None
assert alias_cli is not None
print(json.dumps({
    "omegaprompt": str(pathlib.Path(omegaprompt.__file__).resolve()),
    "omegacal": str(pathlib.Path(omegacal.__file__).resolve()),
    "version": omegaprompt.__version__,
}, sort_keys=True))
"""


def _assert_installed_from_venv(venv_dir: Path, module_json: str) -> dict[str, object]:
    payload = json.loads(module_json)
    prefix = venv_dir.resolve()
    for key in ("omegaprompt", "omegacal"):
        module_path = Path(str(payload[key])).resolve()
        if prefix not in module_path.parents:
            raise SmokeFailure(
                "SMOKE_FAILED",
                f"{key} imported from {module_path}, not the smoke virtualenv {prefix}.",
                details=payload,
            )
    return payload


def _inspect_wheel_examples(wheel: Path) -> dict[str, bool]:
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
    return {
        "examples_shipped_in_wheel": any(name.startswith("examples/") for name in names),
        "reference_artifact_shipped_in_wheel": (
            "examples/reference/reference_artifact.json" in names
        ),
    }


def _sdist_config_includes_examples(pyproject_path: Path = ROOT / "pyproject.toml") -> bool:
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    include = (
        pyproject.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("sdist", {})
        .get("include", [])
    )
    return "examples" in include


def _resolve_reference_artifact(artifact: str | None) -> tuple[Path, str]:
    if artifact:
        path = Path(artifact).resolve()
        source = "explicit"
    else:
        path = REFERENCE_ARTIFACT.resolve()
        source = "repository"
    if not path.exists():
        raise SmokeFailure(
            "ENVIRONMENT_BLOCKED",
            "Reference artifact is not available. The wheel intentionally does not ship examples; run from the repository or pass --artifact.",
            details={"expected": str(path)},
        )
    return path, source


def _smoke_core(venv_dir: Path, wheel: Path, artifact: str | None) -> SmokeResult:
    _install_wheel(venv_dir, wheel)
    python = _python_in(venv_dir)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    proc = _run(
        [python, "-c", _blocked_mcp_import_code()],
        cwd=venv_dir,
        env=env,
        classification="TOOLING_MISSING",
    )
    import_payload = _assert_installed_from_venv(venv_dir, proc.stdout.strip())

    primary_cli = _script_in(venv_dir, "omegaprompt")
    alias_cli = _script_in(venv_dir, "omegacal")
    _run([primary_cli, "--version"], cwd=venv_dir, env=env)
    _run([alias_cli, "--help"], cwd=venv_dir, env=env)

    reference_artifact, artifact_source = _resolve_reference_artifact(artifact)
    report_proc = _run(
        [primary_cli, "report", str(reference_artifact)],
        cwd=venv_dir,
        env=env,
    )
    if "omegaprompt calibration" not in report_proc.stdout:
        raise SmokeFailure(
            "SMOKE_FAILED",
            "omegaprompt report did not render the expected artifact markdown.",
            details={"stdout": report_proc.stdout},
        )

    details: dict[str, object] = {
        **_inspect_wheel_examples(wheel),
        "examples_shipped_in_sdist_config": _sdist_config_includes_examples(),
        "reference_artifact_source": artifact_source,
        "import_payload": import_payload,
    }
    return SmokeResult("core", "OK", "core wheel smoke passed", details)


def _mcp_import_code() -> str:
    expected = sorted(EXPECTED_MCP_TOOLS)
    return f"""
import asyncio
import json
from omegaprompt.mcp import mcp_app

tools = asyncio.run(mcp_app.list_tools())
names = sorted(tool.name for tool in tools)
assert names == {expected!r}, names
print(json.dumps({{"tools": names}}, sort_keys=True))
"""


def _smoke_mcp(venv_dir: Path, wheel: Path) -> SmokeResult:
    _install_wheel(venv_dir, wheel, extra="mcp")
    python = _python_in(venv_dir)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    _run([python, "-m", "omegaprompt.mcp", "--help"], cwd=venv_dir, env=env)
    _run([_script_in(venv_dir, "omegaprompt-mcp"), "--help"], cwd=venv_dir, env=env)
    try:
        proc = _run(
            [python, "-c", _mcp_import_code()],
            cwd=venv_dir,
            env=env,
            classification="TOOLING_MISSING",
        )
    except SmokeFailure as exc:
        stderr = str(exc.details.get("stderr", ""))
        if "TOOLING_MISSING" in stderr or "No module named 'mcp'" in stderr:
            raise SmokeFailure(
                "TOOLING_MISSING",
                "MCP extra smoke requires the mcp SDK in the smoke environment.",
                details=exc.details,
            ) from exc
        raise
    details = json.loads(proc.stdout.strip())
    return SmokeResult("mcp", "OK", "mcp wheel smoke passed", details)


def run_smoke(
    *,
    wheel: Path,
    mode: Literal["core", "mcp", "all"],
    artifact: str | None = None,
    keep_tmp: bool = False,
) -> list[SmokeResult]:
    tmp_obj = tempfile.TemporaryDirectory(prefix="omegaprompt-wheel-smoke-")
    tmp_root = Path(tmp_obj.name)
    results: list[SmokeResult] = []
    try:
        if mode in {"core", "all"}:
            results.append(_smoke_core(_create_venv(tmp_root, "core"), wheel, artifact))
        if mode in {"mcp", "all"}:
            results.append(_smoke_mcp(_create_venv(tmp_root, "mcp"), wheel))
    finally:
        if keep_tmp:
            print(f"kept temp directory: {tmp_root}")
        else:
            tmp_obj.cleanup()
    return results


def _render_results(wheel: Path, results: list[SmokeResult]) -> str:
    lines = [
        "omegaprompt wheel smoke report",
        f"wheel: {wheel}",
        "",
    ]
    for result in results:
        lines.append(f"[{result.classification}] {result.name}: {result.message}")
        for key, value in sorted(result.details.items()):
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test an omegaprompt wheel offline.")
    parser.add_argument(
        "--wheel",
        nargs="+",
        help="Wheel path or glob. If omitted, runs `python -m build --wheel` first.",
    )
    parser.add_argument(
        "--mode",
        choices=["core", "mcp", "all"],
        default="all",
        help="Which smoke environment to run.",
    )
    parser.add_argument(
        "--artifact",
        help="Reference artifact path for the core `omegaprompt report` smoke.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep temporary virtualenvs for debugging.",
    )
    args = parser.parse_args(argv)

    try:
        wheel = _resolve_wheel(args.wheel)
        results = run_smoke(
            wheel=wheel,
            mode=args.mode,
            artifact=args.artifact,
            keep_tmp=args.keep_tmp,
        )
    except SmokeFailure as exc:
        payload = {
            "classification": exc.classification,
            "message": str(exc),
            "details": exc.details,
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"{exc.classification}: {exc}", file=sys.stderr)
            if exc.details:
                print(json.dumps(exc.details, indent=2, sort_keys=True), file=sys.stderr)
        return 2 if exc.classification in {"TOOLING_MISSING", "ENVIRONMENT_BLOCKED"} else 1

    if args.json:
        print(
            json.dumps(
                {
                    "wheel": str(wheel),
                    "results": [
                        {
                            "name": r.name,
                            "classification": r.classification,
                            "message": r.message,
                            "details": r.details,
                        }
                        for r in results
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(_render_results(wheel, results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
