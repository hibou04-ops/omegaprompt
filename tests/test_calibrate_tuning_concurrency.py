"""C2 config + CLI plumbing: CalibrateTuning.max_workers and --concurrency.

CalibrateTuning(max_workers=N) validates, rejects unknown fields (extra=forbid),
rejects values below 1, and the CLI --concurrency flag threads the value through
to a completed calibrate() run (artifact created, exit 0). No wall-clock
assertion — only that the path is wired and stays correct.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from omegaprompt.cli import app
from omegaprompt.core.artifact import load_artifact
from omegaprompt.domain.judge import JudgeResult
from omegaprompt.providers.base import (
    CapabilityTier,
    ProviderCapabilities,
    ProviderResponse,
)
from omegaprompt.runtime import CalibrateTuning


runner = CliRunner()


def test_calibrate_tuning_accepts_max_workers():
    tuning = CalibrateTuning(max_workers=4)
    assert tuning.max_workers == 4


def test_calibrate_tuning_default_max_workers_is_serial():
    assert CalibrateTuning().max_workers == 1


def test_calibrate_tuning_rejects_below_one():
    with pytest.raises(ValidationError):
        CalibrateTuning(max_workers=0)


def test_calibrate_tuning_forbids_unknown_field():
    with pytest.raises(ValidationError):
        CalibrateTuning(max_workers=4, bogus=1)


def _fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    train = tmp_path / "train.jsonl"
    train.write_text(
        "\n".join(json.dumps({"id": f"t{i}", "input": f"task {i}"}) for i in range(4)) + "\n",
        encoding="utf-8",
    )
    rubric = tmp_path / "rubric.json"
    rubric.write_text(
        json.dumps(
            {
                "dimensions": [{"name": "accuracy", "description": "r", "weight": 1.0}],
                "hard_gates": [
                    {"name": "no_refusal", "description": "a", "evaluator": "judge"}
                ],
            }
        ),
        encoding="utf-8",
    )
    variants = tmp_path / "variants.json"
    variants.write_text(
        json.dumps(
            {
                "system_prompts": [
                    "You are a helpful assistant.",
                    "You are a terse senior engineer.",
                ],
                "few_shot_examples": [{"input": "1+1=", "output": "2"}],
            }
        ),
        encoding="utf-8",
    )
    return train, rubric, variants


def test_cli_concurrency_flag_threads_through(monkeypatch, tmp_path):
    """--concurrency N runs a full calibrate() to completion with N workers."""
    from omegaprompt.judges import llm_judge as llm_judge_mod

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")

    class StubProvider:
        name = "anthropic"
        model = "stub-model"

        def capabilities(self):
            return ProviderCapabilities(
                provider=self.name,
                tier=CapabilityTier.CLOUD,
                supports_strict_schema=True,
                supports_json_object=True,
                supports_reasoning_profiles=True,
                supports_usage_accounting=True,
                supports_llm_judge=True,
                ship_grade_judge=True,
            )

        def call(self, request):
            return ProviderResponse(
                text=f"resp:{len(request.system_prompt)}:{len(request.few_shots)}",
                usage={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            )

    monkeypatch.setattr(
        "omegaprompt.runtime.make_provider",
        lambda name, model=None, api_key=None, base_url=None, **_: StubProvider(),
    )

    def fake_score(self, *, rubric, item, target_response):
        parts = target_response.split(":")
        sp_len = int(parts[1]) if len(parts) >= 2 else 0
        fs = int(parts[2]) if len(parts) >= 3 else 0
        score = 1 + min(4, (sp_len // 15) + fs)
        return (
            JudgeResult(scores={"accuracy": score}, gate_results={"no_refusal": True}, notes="x"),
            {
                "input_tokens": 20,
                "output_tokens": 10,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        )

    monkeypatch.setattr(llm_judge_mod.LLMJudge, "score", fake_score)

    train, rubric, variants = _fixtures(tmp_path)
    output_path = tmp_path / "outcome.json"
    result = runner.invoke(
        app,
        [
            "calibrate",
            str(train),
            "--rubric", str(rubric),
            "--variants", str(variants),
            "--output", str(output_path),
            "--unlock-k", "1",
            "--concurrency", "4",
            "--no-fail-on-gate",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, f"CLI failed: {result.stdout}"
    assert output_path.exists()
    artifact = load_artifact(output_path)
    assert artifact.total_api_calls > 0
