# Code Review

Example wording and file labels were edited for neutral software terminology; archived numeric results are historical measurements of their original versioned inputs and were not re-measured after these edits. They do not measure the edited examples. [Preserved prior-source snapshot](https://github.com/hibou04-ops/omegaprompt/tree/v2.2.5/examples/code_review).

## Purpose

Prompt-calibration fixture for code-review responses that prioritize real bugs,
safety issues, and actionable fixes.

## Expected input

`train.jsonl` and optional `test.jsonl` contain review tasks with references.
`rubric.json` defines scoring dimensions and hard gates. `variants*.json`
contains candidate prompt variants.

## Expected output

Calibration outputs are `CalibrationArtifact` JSON files. Historical evaluation
logs/results and subagent captures show previous live-provider experiments and
are not part of default CI.
