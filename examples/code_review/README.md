# Code Review

## Purpose

Prompt-calibration fixture for code-review responses that prioritize real bugs,
safety issues, and actionable fixes.

## Expected input

`train.jsonl` and optional `test.jsonl` contain review tasks with references.
`rubric.json` defines scoring dimensions and hard gates. `variants*.json`
contains candidate prompt variants.

## Expected output

Calibration outputs are `CalibrationArtifact` JSON files. Historical backtest
logs/results and subagent captures show previous live-provider experiments and
are not part of default CI.
