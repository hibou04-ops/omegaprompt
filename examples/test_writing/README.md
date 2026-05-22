# Test Writing

## Purpose

Prompt-calibration fixture for generating useful pytest tests with valid syntax
and meaningful assertions.

## Expected input

`train.jsonl` contains test-writing tasks and references. `rubric.json` defines
coverage, correctness, and hard gates. `variants.json` contains candidate prompt
variants.

## Expected output

`quality_review.json`, `claude_results.json`, and subagent captures are
historical live-provider outputs. New calibration runs should write a
`CalibrationArtifact` JSON.
