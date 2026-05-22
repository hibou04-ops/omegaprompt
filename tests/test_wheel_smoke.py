from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tools import wheel_smoke


def _fake_wheel(path: Path, names: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name in names:
            zf.writestr(name, "{}")
    return path


def test_expand_wheels_accepts_powershell_style_glob(tmp_path: Path) -> None:
    wheel = _fake_wheel(tmp_path / "omegaprompt-1.7.4-py3-none-any.whl", ["x"])

    found = wheel_smoke._expand_wheels([str(tmp_path / "*.whl")])

    assert found == [wheel.resolve()]


def test_inspect_wheel_examples_documents_wheel_boundary(tmp_path: Path) -> None:
    wheel = _fake_wheel(
        tmp_path / "omegaprompt-1.7.4-py3-none-any.whl",
        [
            "omegaprompt/__init__.py",
            "omegacal/__init__.py",
        ],
    )

    info = wheel_smoke._inspect_wheel_examples(wheel)

    assert info == {
        "examples_shipped_in_wheel": False,
        "reference_artifact_shipped_in_wheel": False,
    }
    assert wheel_smoke._sdist_config_includes_examples() is True


def test_inspect_wheel_examples_detects_reference_artifact_if_packaged(tmp_path: Path) -> None:
    wheel = _fake_wheel(
        tmp_path / "omegaprompt-1.7.4-py3-none-any.whl",
        ["examples/reference/reference_artifact.json"],
    )

    info = wheel_smoke._inspect_wheel_examples(wheel)

    assert info["examples_shipped_in_wheel"] is True
    assert info["reference_artifact_shipped_in_wheel"] is True


def test_resolve_reference_artifact_reports_environment_blocked_for_missing_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(wheel_smoke.SmokeFailure) as exc_info:
        wheel_smoke._resolve_reference_artifact(str(tmp_path / "missing.json"))

    assert exc_info.value.classification == "ENVIRONMENT_BLOCKED"
    assert "wheel intentionally does not ship examples" in str(exc_info.value)


def test_create_venv_failure_is_tooling_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class BrokenBuilder:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def create(self, path: Path) -> None:
            raise RuntimeError("venv unavailable")

    monkeypatch.setattr(wheel_smoke.venv, "EnvBuilder", BrokenBuilder)

    with pytest.raises(wheel_smoke.SmokeFailure) as exc_info:
        wheel_smoke._create_venv(tmp_path, "core")

    assert exc_info.value.classification == "TOOLING_MISSING"
    assert "venv unavailable" in str(exc_info.value)


def test_install_wheel_uses_offline_no_deps_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    venv_dir = tmp_path / "venv"
    scripts = venv_dir / ("Scripts" if wheel_smoke.os.name == "nt" else "bin")
    scripts.mkdir(parents=True)
    python = scripts / ("python.exe" if wheel_smoke.os.name == "nt" else "python")
    python.write_text("", encoding="utf-8")
    wheel = tmp_path / "omegaprompt-1.7.4-py3-none-any.whl"
    wheel.write_text("", encoding="utf-8")
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = [str(part) for part in cmd]
        class Proc:
            returncode = 0
            stdout = ""
            stderr = ""
        return Proc()

    monkeypatch.setattr(wheel_smoke, "_run", fake_run)

    wheel_smoke._install_wheel(venv_dir, wheel, extra="mcp")

    cmd = captured["cmd"]
    assert "install" in cmd
    assert "--no-index" in cmd
    assert "--no-deps" in cmd
    assert f"{wheel}[mcp]" in cmd


def test_current_dependency_paths_exclude_repository_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        wheel_smoke.site,
        "getsitepackages",
        lambda: [str(wheel_smoke.ROOT), str(wheel_smoke.ROOT / ".venv" / "Lib" / "site-packages")],
    )
    monkeypatch.setattr(
        wheel_smoke.site,
        "getusersitepackages",
        lambda: str(wheel_smoke.ROOT / "not-a-site"),
    )

    paths = wheel_smoke._current_dependency_paths()

    assert wheel_smoke.ROOT not in paths


def test_blocked_mcp_import_code_blocks_mcp_but_imports_core_shape() -> None:
    code = wheel_smoke._blocked_mcp_import_code()

    assert "class BlockMcp" in code
    assert 'fullname == "mcp"' in code
    assert "import omegaprompt" in code
    assert "import omegacal" in code


def test_main_returns_tooling_missing_when_build_package_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_resolve_wheel(patterns):
        raise wheel_smoke.SmokeFailure("TOOLING_MISSING", "build missing")

    monkeypatch.setattr(wheel_smoke, "_resolve_wheel", fake_resolve_wheel)

    exit_code = wheel_smoke.main(["--mode", "core"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "TOOLING_MISSING: build missing" in captured.err
