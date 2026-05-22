# Refactoring

## Purpose

Prompt-calibration fixture for refactoring guidance that preserves behavior and
keeps changes scoped.

## Expected input

`train.jsonl` contains refactoring tasks and references. `rubric.json` defines
behavior-preservation and actionability expectations. `variants.json` contains
candidate prompt variants.

## Expected output

`quality_review.json`, `claude_results.json`, and subagent captures are
historical live-provider outputs. New calibration runs should write a
`CalibrationArtifact` JSON.
