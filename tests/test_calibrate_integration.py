"""End-to-end calibrate-command integration against real omega_lock.run_p1.

These tests do NOT mock ``omega_lock`` — they run a full P1 cycle with a
deterministic in-memory ``CalibrableTarget`` (no LLM calls). The point is
to catch seams between omega_lock's ``P1Result`` shape and our
``commands/calibrate.py`` result parsing. omega_lock is a hard runtime
dep so this runs in CI without network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from omegaprompt.cli import app
from omegaprompt.core.artifact import load_artifact


runner = CliRunner()


@pytest.fixture
def example_dataset_path(tmp_path: Path) -> Path:
    p = tmp_path / "train.jsonl"
    p.write_text(
        "\n".join(
            json.dumps({"id": f"t{i}", "input": f"task {i}"})
            for i in range(4)
        ) + "\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def example_test_path(tmp_path: Path) -> Path:
    p = tmp_path / "test.jsonl"
    p.write_text(
        "\n".join(
            json.dumps({"id": f"t{i}", "input": f"task {i} (test)"})
            for i in range(3)
        ) + "\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def example_rubric_path(tmp_path: Path) -> Path:
    p = tmp_path / "rubric.json"
    p.write_text(
        json.dumps(
            {
                "dimensions": [
                    {"name": "accuracy", "description": "is it right", "weight": 1.0},
                ],
                "hard_gates": [
                    {"name": "no_refusal", "description": "did it attempt", "evaluator": "judge"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return p


@pytest.fixture
def example_variants_path(tmp_path: Path) -> Path:
    p = tmp_path / "variants.json"
    p.write_text(
        json.dumps(
            {
                "system_prompts": [
                    "You are a helpful assistant.",
                    "You are a terse senior engineer.",
                ],
                "few_shot_examples": [
                    {"input": "1+1=", "output": "2"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return p


def test_calibrate_result_parsing_against_real_p1(
    monkeypatch,
    tmp_path,
    example_dataset_path,
    example_test_path,
    example_rubric_path,
    example_variants_path,
):
    """Drive the real run_p1 with a deterministic target, then verify the
    artifact our CLI writes has the correct best_params / walk_forward /
    sensitivity_ranking shape.

    Strategy: monkeypatch ``make_provider`` to return a stub provider, and
    monkeypatch ``LLMJudge.score`` so the target's ``evaluate`` never issues
    network calls. The fitness function is deterministic on params so
    run_p1 produces a known-good P1Result we can assert against.
    """
    from omegaprompt.commands import calibrate as calibrate_mod
    from omegaprompt.domain.judge import JudgeResult
    from omegaprompt.providers.base import ProviderResponse

    # Ensure env-check in calibrate() doesn't exit 2 when no real key is set.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")

    # Stub provider with deterministic-by-system-prompt response lengths.
    class StubProvider:
        name = "anthropic"
        model = "stub-model"

        def call(self, request):
            # Longer system prompt -> slightly different response.
            response_text = f"resp:{len(request.system_prompt)}:{len(request.few_shots)}"
            return ProviderResponse(
                text=response_text,
                usage={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            )

    def fake_make_provider(name, model=None, api_key=None, base_url=None, **_):
        return StubProvider()

    monkeypatch.setattr(calibrate_mod, "make_provider", fake_make_provider)

    # Deterministic judge: score strictly based on the integer encoded in the
    # target's response text. This gives omega_lock a clean signal per (params,
    # item) pair without any LLM dependency.
    from omegaprompt.judges import llm_judge as llm_judge_mod

    def fake_score(self, *, rubric, item, target_response):
        # target_response = f"resp:{len(system_prompt)}:{len(few_shots)}"
        parts = target_response.split(":")
        sp_len = int(parts[1]) if len(parts) >= 2 else 0
        fs_count = int(parts[2]) if len(parts) >= 3 else 0
        # Reward long system prompt + 1 few-shot.
        score = 1 + min(4, (sp_len // 15) + fs_count)
        return (
            JudgeResult(
                scores={"accuracy": score},
                gate_results={"no_refusal": True},
                notes="stub",
            ),
            {
                "input_tokens": 20,
                "output_tokens": 10,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        )

    monkeypatch.setattr(llm_judge_mod.LLMJudge, "score", fake_score)

    output_path = tmp_path / "outcome.json"
    result = runner.invoke(
        app,
        [
            "calibrate",
            str(example_dataset_path),
            "--rubric", str(example_rubric_path),
            "--variants", str(example_variants_path),
            "--test", str(example_test_path),
            "--output", str(output_path),
            "--unlock-k", "1",
            "--method", "p1",
            "--max-gap", "0.99",
            "--min-kc4", "-1.0",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, f"CLI failed: {result.stdout}"
    assert output_path.exists(), f"artifact not written. stdout={result.stdout}"

    artifact = load_artifact(output_path)

    # best_params is the UNLOCKED params dict from run_p1's grid_best.
    # With one axis unlocked by --unlock-k 1, it should be a 1-entry dict.
    assert isinstance(artifact.best_params, dict)
    assert len(artifact.best_params) >= 1

    # best_fitness pulled from grid_best["fitness"].
    assert artifact.best_fitness > 0

    # walk_forward block populated from top-level walk_forward["test_fitnesses"][0].
    assert artifact.walk_forward is not None
    assert artifact.walk_forward.test_fitness > 0
    assert artifact.walk_forward.train_best_fitness == pytest.approx(artifact.best_fitness)

    # sensitivity_ranking populated from stress_results (not a non-existent "stress" field).
    assert len(artifact.sensitivity_ranking) >= 1
    for row in artifact.sensitivity_ranking:
        assert "axis" in row
        assert "gini_delta" in row
        assert "rank" in row
        assert row["axis"] is not None
        # axis name is a meta-axis name from PromptTarget.param_space()
        assert row["axis"] in {
            "system_prompt_idx",
            "few_shot_count",
            "reasoning_profile_idx",
            "output_budget_idx",
            "response_schema_mode_idx",
            "tool_policy_idx",
        }

    # Provider metadata is always set.
    assert artifact.target_provider == "anthropic"
    assert artifact.target_model == "stub-model"
    assert artifact.judge_provider == "anthropic"

    # total_api_calls reflects target + judge roundtrips.
    assert artifact.total_api_calls > 0

    # status resolves (OK or FAIL_KC4_GATE depending on the gate math).
    assert artifact.status in {"OK", "FAIL_KC4_GATE"}


def test_calibrate_without_test_target_skips_walk_forward(
    monkeypatch,
    tmp_path,
    example_dataset_path,
    example_rubric_path,
    example_variants_path,
):
    """Without --test, the artifact should have walk_forward=None and status=OK."""
    from omegaprompt.commands import calibrate as calibrate_mod
    from omegaprompt.domain.judge import JudgeResult
    from omegaprompt.judges import llm_judge as llm_judge_mod
    from omegaprompt.providers.base import ProviderResponse

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")

    class StubProvider:
        name = "anthropic"
        model = "stub-model"

        def call(self, request):
            return ProviderResponse(
                text="ok",
                usage={"input_tokens": 5, "output_tokens": 2, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
            )

    monkeypatch.setattr(
        calibrate_mod,
        "make_provider",
        lambda name, model=None, api_key=None, base_url=None, **_: StubProvider(),
    )
    monkeypatch.setattr(
        llm_judge_mod.LLMJudge,
        "score",
        lambda self, *, rubric, item, target_response: (
            JudgeResult(scores={"accuracy": 4}, gate_results={"no_refusal": True}),
            {"input_tokens": 10, "output_tokens": 5, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        ),
    )

    output_path = tmp_path / "outcome.json"
    result = runner.invoke(
        app,
        [
            "calibrate",
            str(example_dataset_path),
            "--rubric", str(example_rubric_path),
            "--variants", str(example_variants_path),
            "--output", str(output_path),
            "--unlock-k", "1",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout
    artifact = load_artifact(output_path)
    assert artifact.walk_forward is None
    assert artifact.status == "OK"
