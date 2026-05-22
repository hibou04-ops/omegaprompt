# Summarization

## Purpose

Prompt-calibration fixture for summaries that preserve names, numbers, dates,
decisions, and action items.

## Expected input

`train.jsonl` contains summarization tasks and references. `rubric.json`
defines factual coverage and concision expectations. `variants.json` contains
candidate prompt variants.

## Expected output

`quality_review.json`, `claude_results.json`, and subagent captures are
historical live-provider outputs. New calibration runs should write a
`CalibrationArtifact` JSON.
