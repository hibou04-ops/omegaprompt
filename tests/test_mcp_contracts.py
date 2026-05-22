"""MCP contract tests that bind tools to runtime semantics.

These tests use runtime stubs and the installed MCP SDK only. They never start
the server transport and never call live providers.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from omegaprompt import runtime
from omegaprompt.domain.judge import JudgeResult
from omegaprompt.domain.result import CalibrationArtifact, EvalResult
from omegaprompt.preflight.contracts import (
    AnalyticalFinding,
    PreflightReport,
    PreflightSeverity,
    PreflightStatus,
)

pytest.importorskip("mcp")


EXPECTED_MCP_TOOLS = {
    "calibrate",
    "evaluate",
    "report",
    "diff",
    "measure_sensitivity",
    "grade",
    "preflight",
    "classify_traps",
}


@pytest.fixture
def mcp_server():
    from omegaprompt.mcp import server as srv

    return srv


def _artifact() -> CalibrationArtifact:
    return CalibrationArtifact(
        method="contract",
        unlock_k=0,
        best_params={"system_prompt_variant": 0},
        calibrated_params={"system_prompt_variant": 0},
        best_fitness=0.8,
        calibrated_fitness=0.8,
        neutral_fitness=0.7,
        hard_gate_pass_rate=1.0,
        n_candidates_evaluated=1,
        total_api_calls=0,
        status="OK",
        rationale="contract fixture",
    )


def _eval_result() -> EvalResult:
    return EvalResult(
        params={"system_prompt_variant": 0},
        resolved_params={"system_prompt_variant": 0},
        item_results=[],
        fitness=0.8,
        n_trials=0,
        hard_gate_pass_rate=1.0,
    )


def _json_roundtrip(value):
    return json.loads(json.dumps(value))


def test_mcp_tool_names_match_runtime_exports_and_registered_tools(mcp_server) -> None:
    tools = asyncio.run(mcp_server.mcp_app.list_tools())
    registered = {tool.name for tool in tools}
    runtime_exports = set(runtime.__all__)

    assert registered == EXPECTED_MCP_TOOLS
    assert registered <= runtime_exports
    for name in EXPECTED_MCP_TOOLS:
        assert callable(getattr(mcp_server, name))
        assert callable(getattr(runtime, name))


def test_mcp_imports_cleanly_when_extra_is_installed() -> None:
    from omegaprompt.mcp import mcp_app

    tools = asyncio.run(mcp_app.list_tools())
    assert {tool.name for tool in tools} == EXPECTED_MCP_TOOLS


def test_mcp_calibrate_delegates_and_returns_schema_compatible_artifact(
    mcp_server,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = {}

    def fake_calibrate(**kwargs):
        captured.update(kwargs)
        return _artifact()

    monkeypatch.setattr(mcp_server.runtime, "calibrate", fake_calibrate)

    result = mcp_server.calibrate(
        train=str(tmp_path / "train.jsonl"),
        rubric={"dimensions": [{"name": "quality", "description": "q", "weight": 1.0}]},
        variants={"system_prompts": ["Be precise."]},
        target={"name": "openai", "model": "gpt-test"},
        judge={"name": "anthropic", "model": "claude-test"},
        tuning={"method": "grid", "unlock_k": 1, "profile": "guarded"},
    )

    assert captured["train"].endswith("train.jsonl")
    assert captured["target"] == {"name": "openai", "model": "gpt-test"}
    assert captured["tuning"].method == "grid"
    assert captured["tuning"].unlock_k == 1
    CalibrationArtifact.model_validate(result)
    _json_roundtrip(result)


def test_mcp_evaluate_delegates_and_returns_eval_result_dict(
    mcp_server,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return _eval_result()

    monkeypatch.setattr(mcp_server.runtime, "evaluate", fake_evaluate)

    result = mcp_server.evaluate(
        dataset="dataset.jsonl",
        rubric={"dimensions": [{"name": "quality", "description": "q", "weight": 1.0}]},
        variants={"system_prompts": ["Be precise."]},
        params={"system_prompt_variant": 0},
        target="openai",
        judge="anthropic",
        profile="guarded",
    )

    assert captured["dataset"] == "dataset.jsonl"
    assert captured["params"] == {"system_prompt_variant": 0}
    EvalResult.model_validate(result)
    _json_roundtrip(result)


def test_mcp_report_delegates_and_returns_markdown(
    mcp_server,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_report(artifact):
        captured["artifact"] = artifact
        return "# contract report\n\nok"

    monkeypatch.setattr(mcp_server.runtime, "report", fake_report)

    result = mcp_server.report("artifact.json")

    assert captured["artifact"] == "artifact.json"
    assert isinstance(result, str)
    assert result.startswith("# contract report")
    _json_roundtrip(result)


def test_mcp_diff_delegates_and_preserves_json_or_markdown_format(
    mcp_server,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = []

    def fake_diff(baseline, candidate, *, format):
        captured.append((baseline, candidate, format))
        if format == "markdown":
            return "# omegaprompt diff\n\n## OK"
        return runtime.ArtifactDiff(
            baseline_status="OK",
            candidate_status="OK",
            fitness_delta=0.1,
            neutral_fitness_delta=0.0,
            hard_gate_pass_rate_delta=0.0,
            quality_per_cost_delta=0.0,
            quality_per_latency_delta=0.0,
            regressed=False,
        )

    monkeypatch.setattr(mcp_server.runtime, "diff", fake_diff)

    structured = mcp_server.diff("old.json", "new.json", format="json")
    markdown = mcp_server.diff("old.json", "new.json", format="markdown")

    assert captured == [
        ("old.json", "new.json", "json"),
        ("old.json", "new.json", "markdown"),
    ]
    assert structured["regressed"] is False
    assert isinstance(markdown, str)
    assert markdown.startswith("# omegaprompt diff")
    _json_roundtrip(structured)
    _json_roundtrip(markdown)


def test_mcp_measure_sensitivity_delegates_and_serializes(
    mcp_server,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_measure_sensitivity(**kwargs):
        captured.update(kwargs)
        return runtime.SensitivityResult(
            rows=[{"axis": "reasoning_profile", "normalized_stress": 0.2, "rank": 0}],
            baseline_fitness=0.5,
            n_probes=1,
        )

    monkeypatch.setattr(
        mcp_server.runtime,
        "measure_sensitivity",
        fake_measure_sensitivity,
    )

    result = mcp_server.measure_sensitivity(
        dataset="probe.jsonl",
        rubric={"dimensions": [{"name": "quality", "description": "q", "weight": 1.0}]},
        variants={"system_prompts": ["Be precise."]},
        target="openai",
        judge="anthropic",
        tuning={"profile": "expedition"},
    )

    assert captured["dataset"] == "probe.jsonl"
    assert captured["tuning"].profile.value == "expedition"
    assert result["rows"][0]["axis"] == "reasoning_profile"
    _json_roundtrip(result)


def test_mcp_grade_delegates_and_serializes_judge_result(
    mcp_server,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_grade(**kwargs):
        captured.update(kwargs)
        return runtime.GradeResult(
            judge=JudgeResult(
                scores={"quality": 4},
                gate_results={"correctness": True},
                notes="ok",
            ),
            usage={"input_tokens": 1, "output_tokens": 1},
        )

    monkeypatch.setattr(mcp_server.runtime, "grade", fake_grade)

    result = mcp_server.grade(
        rubric={"dimensions": [{"name": "quality", "description": "q", "weight": 1.0}]},
        item={"id": "case-1", "input": "x", "reference": "y"},
        response="answer",
        provider="anthropic",
        strategy="rule",
    )

    assert captured["strategy"] == "rule"
    assert result["judge"]["scores"]["quality"] == 4
    assert result["usage"]["input_tokens"] == 1
    _json_roundtrip(result)


def test_mcp_preflight_delegates_and_serializes_report(
    mcp_server,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_preflight(**kwargs):
        captured.update(kwargs)
        return PreflightReport(
            status=PreflightStatus.PROCEED,
            warnings=["self-agreement check complete"],
        )

    monkeypatch.setattr(mcp_server.runtime, "preflight", fake_preflight)

    result = mcp_server.preflight(target="openai", judge="anthropic", profile="guarded")

    assert captured == {"target": "openai", "judge": "anthropic", "profile": "guarded"}
    assert result["status"] == "proceed"
    _json_roundtrip(result)


def test_mcp_classify_traps_delegates_and_serializes_findings(
    mcp_server,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_classify_traps(**kwargs):
        captured.update(kwargs)
        return [
            AnalyticalFinding(
                trap_id="self_agreement_bias",
                label="REAL",
                hypothesis="same provider for target and judge",
                severity=PreflightSeverity.MEDIUM,
                note="use cross-vendor judge",
            )
        ]

    monkeypatch.setattr(mcp_server.runtime, "classify_traps", fake_classify_traps)

    result = mcp_server.classify_traps(
        rubric={"dimensions": [{"name": "quality", "description": "q", "weight": 1.0}]},
        variants={"system_prompts": ["Be precise."]},
        target="openai",
        judge="openai",
        dataset="train.jsonl",
        test="test.jsonl",
    )

    assert captured["dataset"] == "train.jsonl"
    assert result[0]["trap_id"] == "self_agreement_bias"
    assert result[0]["severity"] == "medium"
    _json_roundtrip(result)
