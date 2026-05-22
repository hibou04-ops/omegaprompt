# Commit Message

## Purpose

Prompt-calibration fixture for commit messages that preserve the change summary
and useful implementation detail.

## Expected input

`train.jsonl` contains commit-message tasks and references. `rubric.json`
defines scoring dimensions and hard gates. `variants.json` contains candidate
prompt variants.

## Expected output

`quality_review.json`, `claude_results.json`, and subagent captures are
historical live-provider outputs. New calibration runs should write a
`CalibrationArtifact` JSON.
