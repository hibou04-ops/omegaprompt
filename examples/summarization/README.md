# Summarization

Example wording and file labels were edited for neutral software terminology; archived numeric results are historical measurements of their original versioned inputs and were not re-measured after these edits. They do not measure the edited examples. [Preserved prior-source snapshot](https://github.com/hibou04-ops/omegaprompt/tree/v2.2.5/examples/summarization).

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
