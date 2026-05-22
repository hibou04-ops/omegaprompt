# Translation

## Purpose

Prompt-calibration fixture for translations that preserve meaning, tone, and
formatting constraints.

## Expected input

`train.jsonl` contains translation tasks and references. `rubric.json` defines
meaning preservation and format expectations. `variants.json` contains
candidate prompt variants.

## Expected output

`quality_review.json`, `claude_results.json`, and subagent captures are
historical live-provider outputs. New calibration runs should write a
`CalibrationArtifact` JSON.
