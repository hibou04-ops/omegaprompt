"""Reviewer P6: real MCP tool execution tests.

The existing test_mcp_server.py only covers tool *registration* (names,
schemas, descriptions). Reviewer P6 pointed out that this was the
substrate where bugs P1-P4 lived: schema-shape was right, registration
was right, but the actual execution path crashed or returned the wrong
shape. This file calls the registered MCP tool functions directly and
asserts the **executed return value** is well-formed and matches the
underlying runtime semantics.

We patch the providers to keep the tests offline (no API calls) and
exercise the seam where MCP serialisation meets runtime behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omegaprompt.domain import ShipRecommendation
from omegaprompt.domain.result import CalibrationArtifact
from omegaprompt.preflight.contracts import PreflightStatus
from omegaprompt.providers.base import CapabilityTier, ProviderCapabilities


@pytest.fixture
def mcp_server():
    """Import the MCP server module so its decorators run, then return it."""
    from omegaprompt.mcp import server as srv

    return srv


# ---------------------------------------------------------------------------
# grade — proves bug P4#1 stays fixed at the MCP boundary.
# ---------------------------------------------------------------------------


def test_mcp_grade_rule_strategy_returns_serialisable_dict(mcp_server):
    """Pre-fix this would have crashed with AttributeError on tuple."""
    result = mcp_server.grade(
        rubric={
            "dimensions": [{"name": "acc", "description": "is it correct", "weight": 1.0}],
            "hard_gates": [
                {"name": "no_refusal", "description": "must try", "evaluator": "rule"}
            ],
        },
        item={"id": "t1", "input": "ping", "reference": "pong"},
        response="A real and substantive answer that won't trip the refusal heuristic.",
        provider="anthropic",
        strategy="rule",
    )
    # The MCP wrapper must hand the agent a JSON-friendly dict, not a
    # Pydantic object and not a tuple. JSON-encoding it must succeed.
    json.dumps(result)
    assert "judge" in result
    assert "usage" in result
    assert result["judge"]["gate_results"]["no_refusal"] is True


def test_mcp_grade_rule_dict_payload_is_round_trippable(mcp_server):
    """The raw dict the MCP returns must be a CalibrationArtifact-style
    contract: agents will key off scores/gate_results, not unwrap a tuple."""
    result = mcp_server.grade(
        rubric={
            "dimensions": [{"name": "acc", "description": "x", "weight": 1.0}],
            "hard_gates": [],
        },
        item={"id": "t1", "input": "ping"},
        response="A clearly substantive response.",
        provider="anthropic",
        strategy="rule",
    )
    assert "scores" in result["judge"]
    assert "acc" in result["judge"]["scores"]


# ---------------------------------------------------------------------------
# preflight — proves bug P4#3 stays fixed at the MCP boundary.
# ---------------------------------------------------------------------------


def _stub_caps(target_caps, judge_caps, monkeypatch):
    """Patch make_provider + provider_capabilities for offline preflight."""
    class StubProvider:
        def __init__(self, name):
            self.name = name
            self.model = "stub"

    name_to_caps = {"openai": target_caps, "anthropic": judge_caps}
    monkeypatch.setattr(
        "omegaprompt.runtime.make_provider",
        lambda n, model=None, base_url=None: StubProvider(n),
    )
    monkeypatch.setattr(
        "omegaprompt.runtime.provider_capabilities",
        lambda p: name_to_caps[p.name],
    )


def test_mcp_preflight_blocks_placeholder_under_guarded(mcp_server, monkeypatch):
    target_caps = ProviderCapabilities(
        provider="openai", tier=CapabilityTier.CLOUD, placeholder=True,
    )
    judge_caps = ProviderCapabilities(
        provider="anthropic", tier=CapabilityTier.CLOUD,
    )
    _stub_caps(target_caps, judge_caps, monkeypatch)

    result = mcp_server.preflight(target="openai", judge="anthropic", profile="guarded")
    json.dumps(result)
    assert result["status"] == PreflightStatus.ABORT.value
    assert any("placeholder" in r for r in result["blocker_reasons"])


def test_mcp_preflight_proceeds_on_clean_caps(mcp_server, monkeypatch):
    target_caps = ProviderCapabilities(
        provider="openai", tier=CapabilityTier.CLOUD,
    )
    judge_caps = ProviderCapabilities(
        provider="anthropic", tier=CapabilityTier.CLOUD,
    )
    _stub_caps(target_caps, judge_caps, monkeypatch)
    result = mcp_server.preflight(target="openai", judge="anthropic", profile="guarded")
    assert result["status"] == PreflightStatus.PROCEED.value
    assert result["blocker_reasons"] == []


def test_mcp_preflight_self_agreement_warning_visible(mcp_server, monkeypatch):
    caps = ProviderCapabilities(provider="anthropic", tier=CapabilityTier.CLOUD)
    name_to_caps = {"anthropic": caps}

    class StubProvider:
        def __init__(self, name):
            self.name = name
            self.model = "stub"

    monkeypatch.setattr(
        "omegaprompt.runtime.make_provider",
        lambda n, model=None, base_url=None: StubProvider(n),
    )
    monkeypatch.setattr(
        "omegaprompt.runtime.provider_capabilities",
        lambda p: name_to_caps[p.name],
    )
    result = mcp_server.preflight(
        target="anthropic", judge="anthropic", profile="guarded"
    )
    assert any("self-agreement" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# diff — proves bug P2 (status/ship gating) stays fixed at the MCP boundary.
# ---------------------------------------------------------------------------


def _write_artifact(path: Path, **overrides) -> None:
    fields = {
        "method": "p1",
        "unlock_k": 3,
        "best_params": {"system_prompt_variant": 1},
        "best_fitness": 0.8,
        "calibrated_params": {"system_prompt_variant": 1},
        "calibrated_fitness": 0.8,
        "neutral_fitness": 0.5,
        "hard_gate_pass_rate": 1.0,
        "quality_per_cost_best": 0.2,
        "quality_per_latency_best": 0.1,
        "n_candidates_evaluated": 1,
        "total_api_calls": 1,
        "ship_recommendation": ShipRecommendation.SHIP,
        "status": "OK",
    }
    fields.update(overrides)
    artifact = CalibrationArtifact(**fields)
    path.write_text(artifact.model_dump_json(), encoding="utf-8")


def test_mcp_diff_flags_status_failure_as_regression(mcp_server, tmp_path):
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    _write_artifact(old_path, calibrated_fitness=0.6, best_fitness=0.6)
    _write_artifact(
        new_path,
        calibrated_fitness=0.95,  # metrics improved...
        best_fitness=0.95,
        status="FAIL_KC4_GATE",  # ...but status fails
    )
    result = mcp_server.diff(str(old_path), str(new_path), format="json")
    assert result["regressed"] is True
    assert any("status is not OK" in r for r in result["regression_reasons"])


def test_mcp_diff_clean_ship_passes(mcp_server, tmp_path):
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    _write_artifact(old_path, calibrated_fitness=0.6, best_fitness=0.6)
    _write_artifact(new_path, calibrated_fitness=0.85, best_fitness=0.85)
    result = mcp_server.diff(str(old_path), str(new_path), format="json")
    assert result["regressed"] is False


def test_mcp_diff_markdown_format_returns_string(mcp_server, tmp_path):
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    _write_artifact(old_path, calibrated_fitness=0.6, best_fitness=0.6)
    _write_artifact(new_path, calibrated_fitness=0.85, best_fitness=0.85)
    result = mcp_server.diff(str(old_path), str(new_path), format="markdown")
    assert isinstance(result, str)
    assert "omegaprompt diff" in result


# ---------------------------------------------------------------------------
# evaluate — verifies the MCP wrapper hands the loaded artifact through.
#
# The actual `_resolve_params(CalibrationArtifact)` fix from P4#2 is
# regression-tested directly in test_runtime_bug_fixes.py against the
# real runtime path. This test only proves the MCP layer above it: the
# wrapper successfully resolves an artifact-path string into a
# CalibrationArtifact object before calling runtime.evaluate.
# ---------------------------------------------------------------------------


def test_mcp_evaluate_accepts_artifact_path_for_params(mcp_server, tmp_path, monkeypatch):
    """MCP evaluate must unwrap a string artifact path into the artifact
    object that runtime.evaluate expects."""
    # Build a real artifact on disk so the MCP wrapper's load_artifact path runs.
    artifact_path = tmp_path / "art.json"
    _write_artifact(artifact_path, calibrated_fitness=0.7, best_fitness=0.7)

    # Stub the runtime evaluate body so we don't touch the network or
    # the full PromptTarget plumbing — we only need to verify that the
    # MCP wrapper successfully unpacks the artifact path before calling
    # runtime.evaluate.
    captured = {}

    class FakeEvalResult:
        def model_dump(self, mode="json"):
            return {"fitness": 0.7, "n_trials": 1, "params_seen": captured.get("params")}

    def fake_evaluate(*, dataset, rubric, variants, params, target, judge, profile):
        captured["params"] = params
        return FakeEvalResult()

    monkeypatch.setattr("omegaprompt.runtime.evaluate", fake_evaluate)

    result = mcp_server.evaluate(
        dataset="ignored.jsonl",
        rubric={"dimensions": [{"name": "acc", "description": "x", "weight": 1.0}]},
        variants={"system_prompt_variant": [{"id": 0, "value": "x"}]},
        params=str(artifact_path),
        target="anthropic",
    )
    assert result["fitness"] == 0.7
    # Params were unpacked from the artifact path before reaching evaluate:
    assert isinstance(captured["params"], CalibrationArtifact)
    assert captured["params"].calibrated_fitness == 0.7


# ---------------------------------------------------------------------------
# classify_traps — proves the optional-dependency error has a clean shape.
# ---------------------------------------------------------------------------


def test_mcp_classify_traps_missing_dep_raises_importerror(mcp_server, monkeypatch):
    """Reviewer asked for a clean error shape when mini-antemortem-cli is
    not installed. Pre-this-PR the error path was only run when the user
    didn't have mini-antemortem-cli installed; we simulate it via patch."""
    def boom(*a, **kw):
        raise ImportError(
            "omegaprompt.classify_traps requires mini-antemortem-cli."
        )

    monkeypatch.setattr("omegaprompt.runtime.classify_traps", boom)

    with pytest.raises(ImportError, match="mini-antemortem-cli"):
        mcp_server.classify_traps(
            rubric={"dimensions": [{"name": "acc", "description": "x", "weight": 1.0}]},
            variants={"system_prompt_variant": [{"id": 0, "value": "x"}]},
            target="anthropic",
            judge="openai",
            dataset="train.jsonl",
        )


# ---------------------------------------------------------------------------
# report — sanity check that the markdown returner survives unchanged.
# ---------------------------------------------------------------------------


def test_mcp_report_renders_markdown_string(mcp_server, tmp_path):
    artifact_path = tmp_path / "art.json"
    _write_artifact(artifact_path, calibrated_fitness=0.7, best_fitness=0.7)
    md = mcp_server.report(str(artifact_path))
    assert isinstance(md, str)
    assert "omegaprompt calibration" in md
