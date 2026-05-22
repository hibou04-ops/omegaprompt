# Code Writing

## Purpose

Prompt-calibration fixture for writing minimal, correct code without invented
APIs or unnecessary scaffolding.

## Expected input

`train.jsonl` contains code-writing tasks and references. `rubric.json` defines
correctness and output-format expectations. `variants.json` contains candidate
system prompts and few-shot examples.

## Expected output

`quality_review.json`, `claude_results.json`, and subagent captures are
historical live-provider outputs. New calibration runs should write a
`CalibrationArtifact` JSON.
