"""Paced replay of the omegaprompt demo for screencast recording.

Reads `_demo_output.txt` (real output of `examples/demo_calibration.py`) and
reprints each line with deliberate pauses so a 60-second video can capture
each phase. The numbers shown are illustrative demo values aligned to the
60-second subtitle script in `docs/demo/omegaprompt-demo.en.srt`.

Total wall time tuned for ~58 seconds. Adjust SECTION_PAUSES to taste.

Usage::

    PYTHONIOENCODING=utf-8 python examples/demo_replay.py

Regenerate the capture before re-recording if you change the demo script::

    PYTHONIOENCODING=utf-8 python examples/demo_calibration.py > examples/_demo_output.txt 2>&1

For higher-quality recording (no scrollback flicker), open a fresh terminal,
size it to 110x40 minimum, set font size 16-18pt, then run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Pacing rules — substring -> pause AFTER printing that line (seconds).
# First match wins. Order matters.
SECTION_PAUSES: list[tuple[str, float]] = [
    # Cue 1 (0:00-0:04, 4s) — hook: prompt collapse
    ("=== omegaprompt demo ===", 0.5),
    ("problem: prompt scored 4.8/5", 0.5),
    ("day 2 in prod: collapses", 2.5),
    # Cue 2 (0:04-0:14, 10s) — 3 inputs + cross-vendor
    ("---- 3 inputs ----", 0.4),
    ("dataset:", 0.8),
    ("rubric:", 0.8),
    ("variants:", 1.5),
    ("---- Providers (cross-vendor)", 0.4),
    ("target: gpt-4o", 1.0),
    ("judge:  claude-opus", 2.5),
    # Cue 3 (0:14-0:22, 8s) — sensitivity 6 axes
    ("Stress probe over 6", 0.5),
    ("axis                   stress", 0.5),
    ("system_prompt_variant", 0.6),
    ("few_shot_count         0.2150", 0.6),
    ("reasoning_profile", 0.7),
    ("output_budget_bucket", 0.5),
    ("response_schema_mode", 0.5),
    ("tool_policy_variant", 0.7),
    ("3 axes carry signal", 3.5),
    # Cue 4 (0:22-0:29, 7s) — grid search 9 combos
    ("Grid search (top-K=3", 0.4),
    ("9 combinations", 0.6),
    ("[1/9] sp=2 fs=1", 0.8),
    ("[4/9] sp=1 fs=2 rp=standard", 0.8),
    ("[7/9] sp=1 fs=2 rp=deliberate", 1.4),  # winner
    ("best train fitness: 0.9250", 3.0),
    # Cue 5 (0:29-0:36, 7s) — walk-forward
    ("Walk-forward replay", 0.4),
    ("replay best", 0.8),
    ("train fitness: 0.9250", 0.8),
    ("test fitness:  0.9180", 1.2),
    ("generalisation gap", 3.5),
    # Cue 6 (0:36-0:42, 6s) — baseline vs calibrated
    ("Baseline vs calibrated", 0.4),
    ("neutral_baseline_params:", 1.2),
    ("calibrated_params:", 1.2),
    ("uplift:", 3.5),
    # Cue 7 (0:42-0:48, 6s) — schema v2.0 artifact
    ("Schema v2.0 artifact", 0.5),
    ('"schema_version": "2.0"', 0.6),
    ('"neutral_baseline_params":', 0.8),
    ('"calibrated_params":', 1.0),
    ('"walk_forward":', 0.7),
    ('"selected_profile":', 1.5),
    # Cue 8 (0:48-0:55, 7s) — preflight integration
    ("Preflight (plug-in", 0.4),
    ("for noisy environments", 0.8),
    ("pip install mini-omega-lock", 1.5),
    ("adapt thresholds", 2.5),
    # Cue 9 (0:55-1:00, 5s) — install
    ("---- Install ----", 0.4),
    ("pip install omegaprompt", 2.0),
    ("Apache 2.0 - 149 tests", 4.0),
]

DEFAULT_PAUSE = 0.05


def _pause_for(line: str) -> float:
    for pattern, pause in SECTION_PAUSES:
        if pattern in line:
            return pause
    return DEFAULT_PAUSE


def main() -> int:
    here = Path(__file__).resolve().parent
    capture = here / "_demo_output.txt"
    if not capture.exists():
        print(f"ERROR: capture missing at {capture}", file=sys.stderr)
        print(
            "Regenerate with: "
            "PYTHONIOENCODING=utf-8 python examples/demo_calibration.py "
            "> examples/_demo_output.txt 2>&1",
            file=sys.stderr,
        )
        return 1

    text = capture.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    started = time.perf_counter()
    for line in lines:
        print(line, flush=True)
        time.sleep(_pause_for(line))

    elapsed = time.perf_counter() - started
    time.sleep(0.5)
    print(f"\n[demo_replay] elapsed: {elapsed:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
