"""Neutral parameters are indices into the configured axis, not global ordinals."""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from omegaprompt import runtime
from omegaprompt.cli import app
from omegaprompt.core import load_artifact
from omegaprompt.domain import Dataset, JudgeRubric, MetaAxisSpace, PromptVariants
from omegaprompt.domain.enums import OutputBudgetBucket, ReasoningProfile
from omegaprompt.domain.judge import JudgeResult
from omegaprompt.preflight import PreflightReport, derive_adaptation_plan
from omegaprompt.providers.base import CapabilityTier, ProviderCapabilities, ProviderResponse
from omegaprompt.targets.prompt_target import PromptTarget


class RecordingProvider:
    """Local responses through the real generation and LLMJudge request paths."""
    def __init__(self, name="openai"):
        self.name, self.model, self.requests = name, "offline-neutral-fixture", []

    def capabilities(self):
        return ProviderCapabilities(
            provider=self.name, tier=CapabilityTier.CLOUD,
            supports_strict_schema=True, supports_json_object=True,
            supports_reasoning_profiles=True, supports_usage_accounting=True,
            supports_llm_judge=True, ship_grade_judge=True,
        )

    def call(self, request):
        self.requests.append(request)
        usage = {"input_tokens": 10, "output_tokens": 2}
        if request.output_schema is not None:
            payload = json.loads(request.user_message)
            score = int(payload["target_response"]["text"].rsplit(":", 1)[1])
            return ProviderResponse(
                parsed=JudgeResult(scores={"quality": score}, gate_results={"valid": True}),
                usage=usage,
            )
        return ProviderResponse(text="score:" + ("5" if len(request.system_prompt) > 20 else "2"),
                                usage=usage)


def inputs():
    train = Dataset.from_items([{"id": f"r{i}", "input": f"training question {i}"}
                                for i in range(4)])
    test = Dataset.from_items([{"id": f"t{i}", "input": f"heldout question {i}"}
                               for i in range(4)])
    rubric = JudgeRubric.model_validate({
        "dimensions": [{"name": "quality", "description": "quality", "weight": 1}],
        "hard_gates": [{"name": "valid", "description": "valid"}],
    })
    variants = PromptVariants(system_prompts=["Answer briefly", "Answer carefully and explain all steps"],
                              few_shot_examples=[])
    return train, test, rubric, variants


@pytest.mark.parametrize("reasoning,budgets,expected_reasoning,expected_budget", [
    (None, None, ReasoningProfile.STANDARD, OutputBudgetBucket.MEDIUM),
    (["standard"], ["medium"], ReasoningProfile.STANDARD, OutputBudgetBucket.MEDIUM),
    (["standard", "deep", "off"], ["medium", "large", "small"],
     ReasoningProfile.STANDARD, OutputBudgetBucket.MEDIUM),
    (["deep", "off", "standard"], ["large", "small", "medium"],
     ReasoningProfile.STANDARD, OutputBudgetBucket.MEDIUM),
    (["off", "deep"], ["large"], ReasoningProfile.OFF, OutputBudgetBucket.LARGE),
    (["deep"], ["small"], ReasoningProfile.DEEP, OutputBudgetBucket.SMALL),
])
def test_neutral_matches_axis_preferences_specs_and_actual_requests(
    reasoning, budgets, expected_reasoning, expected_budget,
):
    from omegaprompt.judges import LLMJudge

    train, _, rubric, variants = inputs()
    provider, judge_provider = RecordingProvider(), RecordingProvider("anthropic")
    space = None if reasoning is None else MetaAxisSpace(
        system_prompt_idx_max=1, few_shot_max=0,
        reasoning_profiles=reasoning, output_budgets=budgets,
    )
    target = PromptTarget(target_provider=provider, judge=LLMJudge(provider=judge_provider),
                          dataset=train, rubric=rubric, variants=variants, space=space)
    neutral = target.neutral_params()
    assert {spec.name: spec.neutral for spec in target.param_space()} == neutral
    assert target.space.reasoning_profiles[neutral["reasoning_profile"]] == expected_reasoning
    assert target.space.output_budgets[neutral["output_budget_bucket"]] == expected_budget
    result = target.evaluate(None)
    assert not result.boundary_warnings
    assert len(provider.requests) == len(train) == len(judge_provider.requests)
    assert all(request.reasoning_profile == expected_reasoning for request in provider.requests)
    assert all(request.output_budget_bucket == expected_budget for request in provider.requests)


def singleton_tuning():
    return runtime.CalibrateTuning(unlock_k=1, validation_mode="disjoint", space=MetaAxisSpace(
        system_prompt_idx_max=1, few_shot_max=0,
        reasoning_profiles=[ReasoningProfile.STANDARD], output_budgets=[OutputBudgetBucket.MEDIUM],
    ))


def failed_measurement_plan():
    return derive_adaptation_plan(report=PreflightReport(
        measurement_status={"judge_consistency": "failed"},
    ))


def assert_subset_artifact(artifact, provider):
    assert artifact.neutral_baseline_params["reasoning_profile"] == "standard"
    assert artifact.neutral_baseline_params["output_budget_bucket"] == "medium"
    assert provider.requests
    assert all(request.reasoning_profile == ReasoningProfile.STANDARD for request in provider.requests)
    assert all(request.output_budget_bucket == OutputBudgetBucket.MEDIUM for request in provider.requests)


@pytest.mark.parametrize("manual_review", [False, True])
def test_runtime_subset_calibration_preserves_manual_review_escalation(tmp_path, manual_review):
    train, test, rubric, variants = inputs()
    target, judge = RecordingProvider(), RecordingProvider("anthropic")
    output = tmp_path / "artifact.json"
    artifact = runtime.calibrate(
        train, test=test, rubric=rubric, variants=variants, target=target, judge=judge,
        tuning=singleton_tuning(), adaptation_plan=failed_measurement_plan() if manual_review else None,
        output=output,
    )
    assert load_artifact(output).model_dump(mode="json") == artifact.model_dump(mode="json")
    assert_subset_artifact(artifact, target)
    assert artifact.status == ("REQUIRES_MANUAL_REVIEW" if manual_review else "OK")
    assert artifact.ship_recommendation == ("hold" if manual_review else "ship")
    if manual_review:
        assert artifact.adaptation_summary["manual_review_required"] is True


def write_cli_inputs(tmp_path):
    train, test, rubric, variants = inputs()
    paths = {name: tmp_path / filename for name, filename in [
        ("train", "train.jsonl"), ("test", "test.jsonl"), ("rubric", "rubric.json"),
        ("variants", "variants.json"), ("space", "space.json"), ("plan", "plan.json"),
    ]}
    for name, dataset in [("train", train), ("test", test)]:
        paths[name].write_text("\n".join(item.model_dump_json() for item in dataset.items) + "\n")
    for name, value in [("rubric", rubric), ("variants", variants),
                        ("space", singleton_tuning().space), ("plan", failed_measurement_plan())]:
        paths[name].write_text(value.model_dump_json())
    return paths


def install_recording_providers(monkeypatch):
    providers = {name: RecordingProvider(name) for name in ("openai", "anthropic")}
    monkeypatch.setattr(runtime, "make_provider", lambda name, **kwargs: providers[name])
    return providers


def test_cli_subset_space_runs_real_runtime_and_keeps_review_hold(tmp_path, monkeypatch):
    paths = write_cli_inputs(tmp_path)
    providers = install_recording_providers(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-fixture")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-fixture")
    output = tmp_path / "cli-artifact.json"
    result = CliRunner().invoke(app, [
        "calibrate", str(paths["train"]), "--test", str(paths["test"]),
        "--rubric", str(paths["rubric"]), "--variants", str(paths["variants"]),
        "--space", str(paths["space"]), "--adaptation-plan", str(paths["plan"]),
        "--target-provider", "openai", "--judge-provider", "anthropic",
        "--validation-mode", "disjoint", "--unlock-k", "1", "--output", str(output),
        "--fail-on-gate",
    ], catch_exceptions=False)
    assert result.exit_code == 1, result.output  # execution completed; review HOLD still blocks CI
    assert output.is_file(), result.output
    artifact = load_artifact(output)
    assert_subset_artifact(artifact, providers["openai"])
    assert artifact.status == "REQUIRES_MANUAL_REVIEW"
    assert artifact.ship_recommendation == "hold"


def test_mcp_subset_tuning_runs_real_runtime_and_keeps_review_hold(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from omegaprompt.mcp import server
    from omegaprompt.domain.result import CalibrationArtifact

    paths = write_cli_inputs(tmp_path)
    providers = install_recording_providers(monkeypatch)
    data = server.calibrate(
        train=str(paths["train"]), test=str(paths["test"]),
        rubric=str(paths["rubric"]), variants=str(paths["variants"]),
        target="openai", judge="anthropic", tuning=singleton_tuning().model_dump(mode="json"),
        adaptation_plan=failed_measurement_plan().model_dump(mode="json"),
    )
    artifact = CalibrationArtifact.model_validate(data)
    assert_subset_artifact(artifact, providers["openai"])
    assert data["status"] == "REQUIRES_MANUAL_REVIEW"
    assert data["ship_recommendation"] == "hold"
