from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def _run_script(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)
    env.pop("GEMINI_API_KEY", None)
    env.pop("GOOGLE_API_KEY", None)
    return subprocess.run(
        [sys.executable, str(EXAMPLES / script), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def test_demo_calibration_output_matches_stable_snapshot() -> None:
    snapshot = (EXAMPLES / "_demo_output.txt").read_text(encoding="utf-8")

    result = _run_script("demo_calibration.py")

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout == snapshot
    assert "release_approved: True" in result.stdout
    assert "strict_blocking_findings: 0" in result.stdout


def test_demo_replay_default_is_byte_stable_and_fast() -> None:
    snapshot = (EXAMPLES / "_demo_output.txt").read_text(encoding="utf-8")

    result = _run_script("demo_replay.py")

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout == snapshot
    assert "elapsed" not in result.stdout.lower()


def test_demo_replay_missing_capture_reports_actionable_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"

    result = _run_script("demo_replay.py", "--capture", str(missing))

    assert result.returncode == 1
    assert "ERROR: capture missing" in result.stderr
    assert "examples/demo_calibration.py" in result.stderr


def test_demo_scripts_do_not_call_live_provider_paths() -> None:
    for script_name in ["demo_calibration.py", "demo_replay.py"]:
        source = (EXAMPLES / script_name).read_text(encoding="utf-8")
        assert "make_provider(" not in source
        assert "ProviderRequest" not in source
        assert "ANTHROPIC_API_KEY" not in source
        assert "OPENAI_API_KEY" not in source
        assert "GEMINI_API_KEY" not in source
        assert "GOOGLE_API_KEY" not in source
    replay_source = (EXAMPLES / "demo_replay.py").read_text(encoding="utf-8")
    assert "check_artifact_integrity" in replay_source


def test_demo_snapshot_contains_no_stale_exact_test_count_claim() -> None:
    snapshot = (EXAMPLES / "_demo_output.txt").read_text(encoding="utf-8")

    assert "149 tests" not in snapshot
    assert "317 passing" not in snapshot
    assert "tests -" not in snapshot
    assert "deterministic in-memory reference providers" in snapshot


def test_examples_gallery_distinguishes_offline_and_live_paths() -> None:
    gallery = (EXAMPLES / "README.md").read_text(encoding="utf-8")

    assert "Offline deterministic demos" in gallery
    assert "Live provider task fixtures" in gallery
    assert "OMEGAPROMPT_LIVE_PROVIDER_TESTS=1" in gallery
    assert "No row above implies provider or model superiority." in gallery


def test_each_examples_task_directory_has_purpose_and_io_shape() -> None:
    task_dirs = [
        path
        for path in EXAMPLES.iterdir()
        if path.is_dir() and ((path / "train.jsonl").exists() or path.name == "reference")
    ]

    assert task_dirs
    for task_dir in task_dirs:
        readme = task_dir / "README.md"
        assert readme.exists(), f"{task_dir} is missing README.md"
        text = readme.read_text(encoding="utf-8")
        assert "## Purpose" in text
        assert "## Expected input" in text
        assert "## Expected output" in text
