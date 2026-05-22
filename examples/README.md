# omegaprompt examples

This gallery separates deterministic offline demos from live provider examples.
Default CI and the replay demo do not call provider APIs.

## Offline deterministic demos

| Path | Purpose | Command | Network/API keys |
| --- | --- | --- | --- |
| `demo_replay.py` | Replay the stable captured demo output. | `PYTHONIOENCODING=utf-8 python examples/demo_replay.py` | none |
| `demo_calibration.py` | Read the checked-in golden artifact, verify integrity, and print artifact-backed metrics. | `PYTHONIOENCODING=utf-8 python examples/demo_calibration.py` | none |
| `reference/reproduce_reference_artifact.py` | Regenerate all golden reference artifacts and the manifest. | `python examples/reference/reproduce_reference_artifact.py` | none |

## Reference artifacts

`examples/reference/` is the measurement-grade offline harness. It uses
deterministic in-memory providers and judges only. Exact metrics in demo output
come from these artifacts and their manifest, not from live provider calls.

Use:

```bash
python examples/reference/reproduce_reference_artifact.py
omegaprompt check-artifact examples/reference/reference_artifact.json --strict
omegaprompt report examples/reference/reference_artifact.json
```

## Live provider task fixtures

The task directories contain datasets, rubrics, variants, and historical output
captures for real prompt-calibration tasks. They are examples of input shape and
review workflow. Running them against Anthropic, OpenAI, Gemini, or a local
OpenAI-compatible endpoint is opt-in and requires provider configuration.

Live provider runs are not part of default CI. Optional live smoke tests must be
enabled explicitly with `OMEGAPROMPT_LIVE_PROVIDER_TESTS=1`.

| Directory | Purpose | Expected input shape | Expected output shape |
| --- | --- | --- | --- |
| `code_review/` | Calibrate code-review prompts against correctness and safety issues. | `train.jsonl`, optional `test.jsonl`, `rubric.json`, `variants*.json`. | `CalibrationArtifact` JSON, backtest logs/results, reviewed outputs. |
| `code_writing/` | Compare code-writing prompt variants for concise, correct implementations. | `train.jsonl`, `rubric.json`, `variants.json`. | `quality_review.json`, `claude_results.json`, subagent prompt/output captures. |
| `commit_message/` | Tune commit-message generation for concise summaries and body detail. | `train.jsonl`, `rubric.json`, `variants.json`. | `quality_review.json`, `claude_results.json`, subagent prompt/output captures. |
| `debugging/` | Tune debugging responses that identify root cause and minimal fixes. | `train.jsonl`, `rubric.json`, `variants.json`. | `quality_review.json`, `claude_results.json`, subagent prompt/output captures. |
| `explanation/` | Tune explanatory answers for clarity and factual coverage. | `train.jsonl`, `rubric.json`, `variants.json`. | `quality_review.json`, `claude_results.json`, subagent prompt/output captures. |
| `refactoring/` | Tune refactoring guidance for behavior preservation and minimal change. | `train.jsonl`, `rubric.json`, `variants.json`. | `quality_review.json`, `claude_results.json`, subagent prompt/output captures. |
| `summarization/` | Tune summarization prompts for preserving names, numbers, and actions. | `train.jsonl`, `rubric.json`, `variants.json`. | `quality_review.json`, `claude_results.json`, subagent prompt/output captures. |
| `test_writing/` | Tune test-writing prompts for real pytest coverage and syntax validity. | `train.jsonl`, `rubric.json`, `variants.json`. | `quality_review.json`, `claude_results.json`, subagent prompt/output captures. |
| `translation/` | Tune translation prompts for meaning preservation and formatting. | `train.jsonl`, `rubric.json`, `variants.json`. | `quality_review.json`, `claude_results.json`, subagent prompt/output captures. |

No row above implies provider or model superiority. The examples are fixtures;
provider suitability is governed by `ProviderCapabilities` and the artifact's
recorded capability events.
