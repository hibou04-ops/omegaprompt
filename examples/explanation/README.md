# Explanation

## Purpose

Prompt-calibration fixture for explanatory answers that are accurate, clear,
and appropriately concise.

## Expected input

`train.jsonl` contains explanation tasks and references. `rubric.json` defines
accuracy, clarity, and hard gates. `variants.json` contains candidate prompt
variants.

## Expected output

`quality_review.json`, `claude_results.json`, and subagent captures are
historical live-provider outputs. New calibration runs should write a
`CalibrationArtifact` JSON.
