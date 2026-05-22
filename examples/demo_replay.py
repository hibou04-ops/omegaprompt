"""Replay the deterministic omegaprompt demo snapshot.

Default mode is byte-stable and fast: it prints ``examples/_demo_output.txt``
exactly, with no sleeping and no wall-clock footer. Pass ``--paced`` when
recording a screencast.

Usage::

    PYTHONIOENCODING=utf-8 python examples/demo_replay.py
    PYTHONIOENCODING=utf-8 python examples/demo_replay.py --paced
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from omegaprompt.core.artifact_integrity import check_artifact_integrity

SECTION_PAUSES: list[tuple[str, float]] = [
    ("=== omegaprompt deterministic offline demo ===", 0.4),
    ("mode: no API keys", 0.5),
    ("reproduce:", 0.7),
    ("---- Inputs ----", 0.4),
    ("dataset:", 0.6),
    ("rubric:", 0.6),
    ("variants:", 1.0),
    ("---- Reference artifact integrity ----", 0.4),
    ("schema_version:", 0.5),
    ("status:", 0.5),
    ("ship_recommendation:", 0.5),
    ("release_approved:", 0.7),
    ("strict_blocking_findings:", 0.7),
    ("normalized_hash:", 1.2),
    ("---- Deterministic metrics from the artifact ----", 0.4),
    ("neutral_fitness:", 0.5),
    ("calibrated_fitness:", 0.5),
    ("uplift_absolute:", 0.5),
    ("uplift_percent:", 0.8),
    ("walk_forward_mode:", 0.5),
    ("test_fitness:", 0.5),
    ("generalization_gap:", 0.5),
    ("kc4_status:", 0.5),
    ("walk_forward_passed:", 0.8),
    ("---- Sensitivity ranking ----", 0.4),
    ("1. system_prompt_variant", 0.6),
    ("2. reasoning_profile", 0.6),
    ("3. few_shot_count", 0.8),
    ("---- Live provider path ----", 0.4),
    ("offline demo:", 0.7),
    ("live examples:", 0.9),
    ("default CI:", 0.7),
    ("gallery:", 0.8),
]

DEFAULT_PAUSE = 0.02


def _verify_reference_artifact(here: Path) -> bool:
    artifact = here / "reference" / "reference_artifact.json"
    report = check_artifact_integrity(artifact)
    if report.valid:
        return True
    print("ERROR: reference artifact failed integrity checks", file=sys.stderr)
    for finding in report.findings:
        print(f"{finding.severity} {finding.id}: {finding.message}", file=sys.stderr)
    return False


def _pause_for(line: str) -> float:
    for pattern, pause in SECTION_PAUSES:
        if pattern in line:
            return pause
    return DEFAULT_PAUSE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay the deterministic demo snapshot.")
    parser.add_argument(
        "--capture",
        default=None,
        help="Path to a captured demo output file. Defaults to examples/_demo_output.txt.",
    )
    parser.add_argument(
        "--paced",
        action="store_true",
        help="Sleep between lines for screencast recording.",
    )
    args = parser.parse_args(argv)

    here = Path(__file__).resolve().parent
    if not _verify_reference_artifact(here):
        return 1

    capture = Path(args.capture) if args.capture else here / "_demo_output.txt"
    if not capture.exists():
        print(f"ERROR: capture missing at {capture}", file=sys.stderr)
        print(
            "Regenerate with: "
            "PYTHONIOENCODING=utf-8 python examples/demo_calibration.py "
            "> examples/_demo_output.txt",
            file=sys.stderr,
        )
        return 1

    lines = capture.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines:
        print(line, flush=True)
        if args.paced:
            time.sleep(_pause_for(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
