# Debugging

## Purpose

Prompt-calibration fixture for debugging answers that identify likely root
cause, evidence, and a minimal fix path.

## Expected input

`train.jsonl` contains debugging tasks and references. `rubric.json` defines
diagnostic quality and hard gates. `variants.json` contains candidate prompt
variants.

## Expected output

`quality_review.json`, `claude_results.json`, and subagent captures are
historical live-provider outputs. New calibration runs should write a
`CalibrationArtifact` JSON.
