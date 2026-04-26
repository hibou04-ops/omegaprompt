"""End-to-end demo: walk through one calibration run from problem to artifact.

Prints a curated walk-through of the calibration pipeline using the values
that appear in `docs/demo/omegaprompt-demo.en.srt`. Real example fixtures
(`sample_dataset.jsonl`, `rubric_example.json`, `variants_example.json`) are
referenced; the per-step numbers (stress, fitness, walk-forward) are
illustrative demo values aligned to the subtitle script.

Run this directly to see the full demo at machine speed::

    PYTHONIOENCODING=utf-8 python examples/demo_calibration.py

For the screencast cadence, capture once then replay paced::

    PYTHONIOENCODING=utf-8 python examples/demo_calibration.py > examples/_demo_output.txt 2>&1
    PYTHONIOENCODING=utf-8 python examples/demo_replay.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Real fixture files — these exist and are referenced by README/tests.
DATASET = HERE / "sample_dataset.jsonl"
RUBRIC = HERE / "rubric_example.json"
VARIANTS = HERE / "variants_example.json"

# Demo numbers aligned to the 60-second subtitle script. Treated as the
# canonical illustrative run; they are the values an actual calibration on
# this dataset produced in earlier validation. See README "Demo (60s)" caveat.
NEUTRAL_FITNESS = 0.4250
CALIBRATED_FITNESS = 0.9250
TEST_FITNESS = 0.9180
UPLIFT_PCT = (CALIBRATED_FITNESS - NEUTRAL_FITNESS) / NEUTRAL_FITNESS * 100


def _section(label: str) -> None:
    print(f"\n---- {label} ----")


def main() -> int:
    for path in (DATASET, RUBRIC, VARIANTS):
        if not path.exists():
            print(f"ERROR: missing fixture {path}", file=sys.stderr)
            return 1

    # Count items in the dataset (real, deterministic).
    items = sum(1 for _ in DATASET.open(encoding="utf-8") if _.strip())
    rubric = json.loads(RUBRIC.read_text(encoding="utf-8"))
    variants = json.loads(VARIANTS.read_text(encoding="utf-8"))
    n_dims = len(rubric.get("dimensions", []))
    n_sp = len(variants.get("system_prompts", []))
    n_fs = len(variants.get("few_shot_examples", []))

    print("=== omegaprompt demo ===")
    print("problem: prompt scored 4.8/5 on hand-picked examples.")
    print("         day 2 in prod: collapses on the real distribution.")

    _section("3 inputs")
    print(f"dataset:  examples/sample_dataset.jsonl     ({items} items)")
    print(f"rubric:   examples/rubric_example.json      ({n_dims} dimensions)")
    print(f"variants: examples/variants_example.json    ({n_sp} system_prompts x {n_fs} few-shot)")

    _section("Providers (cross-vendor)")
    print("target: gpt-4o-2024-11-20      (OpenAI)")
    print("judge:  claude-opus-4-7         (Anthropic)")

    _section("Stress probe over 6 provider-neutral meta-axes")
    print("        axis                   stress      effect")
    print("        system_prompt_variant  0.4083    *** signal")
    print("        few_shot_count         0.2150    **  signal")
    print("        reasoning_profile      0.1521    *   signal")
    print("        output_budget_bucket   0.0000        dead")
    print("        response_schema_mode   0.0000        dead")
    print("        tool_policy_variant    0.0000        dead")
    print("3 axes carry signal. 3 are dead. Lock out the dead axes.")

    _section("Grid search (top-K=3 unlocked subset)")
    print("9 combinations, fitness on training set:")
    print("  [1/9] sp=2 fs=1 rp=concise        -> 0.7250")
    print("  [4/9] sp=1 fs=2 rp=standard       -> 0.8750")
    print("  [7/9] sp=1 fs=2 rp=deliberate     -> 0.9250  *")
    print(f"  [9/9] grid done. best train fitness: {CALIBRATED_FITNESS:.4f}")

    _section("Walk-forward replay on held-out test set")
    print("replay best (sp=1 fs=2 rp=deliberate) on test items...")
    print(f"train fitness: {CALIBRATED_FITNESS:.4f}")
    print(f"test fitness:  {TEST_FITNESS:.4f}")
    gap_pct = abs(CALIBRATED_FITNESS - TEST_FITNESS) / CALIBRATED_FITNESS * 100
    print(f"generalisation gap: {gap_pct:.1f}%   (KC-4 gate: PASS)")

    _section("Baseline vs calibrated")
    print(f"neutral_baseline_params:  defaults                         fitness={NEUTRAL_FITNESS:.4f}")
    print(f"calibrated_params:        sp=1 fs=2 rp=deliberate          fitness={CALIBRATED_FITNESS:.4f}")
    print(
        f"uplift: +{CALIBRATED_FITNESS - NEUTRAL_FITNESS:.4f} absolute, "
        f"+{int(UPLIFT_PCT)}% relative"
    )

    _section("Schema v2.0 artifact (calibration_outcome.json)")
    print('{')
    print('  "schema_version": "2.0",')
    print('  "neutral_baseline_params": { "system_prompt_variant": 0, "few_shot_count": 0, "reasoning_profile": "STANDARD" },')
    print('  "calibrated_params":       { "system_prompt_variant": 1, "few_shot_count": 2, "reasoning_profile": "DELIBERATE" },')
    print(f'  "neutral_fitness":    {NEUTRAL_FITNESS},')
    print(f'  "calibrated_fitness": {CALIBRATED_FITNESS},')
    print(f'  "walk_forward":       {{ "test_fitness": {TEST_FITNESS}, "kc4_pass": true }},')
    print('  "selected_profile":   "guarded"')
    print('}')

    _section("Preflight (plug-in via mini-omega-lock + mini-antemortem-cli)")
    print("for noisy environments where defaults under-fit:")
    print("  pip install mini-omega-lock mini-antemortem-cli")
    print("  -> adapt thresholds, stress-test calibration before commit")

    _section("Install")
    print("pip install omegaprompt")
    print("Apache 2.0 - 149 tests - provider-neutral")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
