"""Sensitivity + artifact round-trip tests."""

from __future__ import annotations

from pathlib import Path

from omegaprompt.core.artifact import load_artifact, save_artifact
from omegaprompt.core.sensitivity import measure_sensitivity, select_unlocked_axes
from omegaprompt.domain.result import CalibrationArtifact
from tests.helpers import workspace_tmpdir


def test_measure_sensitivity_ranks_high_variance_first():
    def evaluate(params: dict) -> float:
        # "a" swings fitness by 10x more than "b"
        return 10.0 * params.get("a", 0) + 0.1 * params.get("b", 0)

    baseline = {"a": 0, "b": 0}
    probes = {
        "a": [{"a": 1, "b": 0}, {"a": 2, "b": 0}, {"a": 3, "b": 0}],
        "b": [{"a": 0, "b": 1}, {"a": 0, "b": 2}, {"a": 0, "b": 3}],
    }
    scores = measure_sensitivity(evaluate, probes, baseline)
    assert scores[0].axis == "a"
    assert scores[0].rank == 0
    assert scores[1].axis == "b"
    assert scores[0].gini_delta >= scores[1].gini_delta


def test_select_unlocked_axes_top_k():
    def evaluate(params: dict) -> float:
        return params.get("a", 0) * 1.0 + params.get("b", 0) * 0.5 + params.get("c", 0) * 0.1

    baseline = {"a": 0, "b": 0, "c": 0}
    probes = {
        "a": [{"a": 1, "b": 0, "c": 0}, {"a": 3, "b": 0, "c": 0}],
        "b": [{"a": 0, "b": 1, "c": 0}, {"a": 0, "b": 3, "c": 0}],
        "c": [{"a": 0, "b": 0, "c": 1}, {"a": 0, "b": 0, "c": 3}],
    }
    scores = measure_sensitivity(evaluate, probes, baseline)
    assert select_unlocked_axes(scores, k=0) == []
    top_two = select_unlocked_axes(scores, k=2)
    assert len(top_two) == 2
    # Top-2 should skip "c" which has the lowest per-delta Gini.
    assert "c" not in top_two or scores[2].axis == "c"


def test_artifact_roundtrip():
    with workspace_tmpdir() as tmp_path:
        artifact = CalibrationArtifact(
            method="p1",
            unlock_k=3,
            best_params={"system_prompt_variant": 1, "reasoning_profile": "deep"},
            best_fitness=0.87,
            neutral_baseline_params={"system_prompt_variant": 0},
            calibrated_params={"system_prompt_variant": 1, "reasoning_profile": "deep"},
            neutral_fitness=0.80,
            calibrated_fitness=0.87,
            hard_gate_pass_rate=1.0,
            sensitivity_ranking=[
                {"axis": "system_prompt_variant", "gini_delta": 0.42, "rank": 0},
                {"axis": "few_shot_count", "gini_delta": 0.18, "rank": 1},
            ],
            n_candidates_evaluated=40,
            total_api_calls=160,
            usage_summary={"input_tokens": 100, "output_tokens": 50},
            target_provider="openai",
            target_model="gpt-4o",
            judge_provider="anthropic",
            judge_model="claude-opus-4-7",
        )
        path = tmp_path / "artifact.json"
        save_artifact(artifact, path)
        loaded = load_artifact(path)
        assert loaded.best_fitness == 0.87
        assert loaded.best_params["system_prompt_variant"] == 1
        assert loaded.sensitivity_ranking[0]["axis"] == "system_prompt_variant"
        assert loaded.target_provider == "openai"
