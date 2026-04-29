"""Tests for omegaprompt.runtime — Tier 1 entrypoints.

Pure-function tests only (report, diff, type coercion). The calibrate
and evaluate entrypoints are covered by test_calibrate_integration and
test_target respectively, since they require provider-stubbed LLM calls.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omegaprompt import (
    ArtifactDiff,
    CalibrateTuning,
    CalibrationArtifact,
    ExecutionProfile,
    ProviderSpec,
    diff,
    report,
)
from omegaprompt.runtime import _resolve_artifact, _resolve_provider


def _make_artifact(
    *,
    calibrated_fitness: float = 0.75,
    hard_gate_pass_rate: float = 0.9,
    status: str = "OK",
) -> CalibrationArtifact:
    return CalibrationArtifact(
        method="p1",
        unlock_k=3,
        best_params={"system_prompt_variant": 1},
        best_fitness=calibrated_fitness,
        neutral_baseline_params={"system_prompt_variant": 0},
        calibrated_params={"system_prompt_variant": 1},
        neutral_fitness=0.50,
        calibrated_fitness=calibrated_fitness,
        hard_gate_pass_rate=hard_gate_pass_rate,
        n_candidates_evaluated=50,
        total_api_calls=500,
        status=status,
    )


# ----- Pydantic public types -----


class TestProviderSpec:
    def test_minimal(self):
        s = ProviderSpec(name="anthropic")
        assert s.name == "anthropic"
        assert s.model is None
        assert s.base_url is None

    def test_full(self):
        s = ProviderSpec(name="openai", model="gpt-4o", base_url="https://api.openai.com/v1")
        assert s.model == "gpt-4o"
        assert s.base_url == "https://api.openai.com/v1"

    def test_extra_keys_rejected(self):
        with pytest.raises(ValidationError):
            ProviderSpec(name="anthropic", invalid_field="x")


class TestCalibrateTuning:
    def test_defaults(self):
        t = CalibrateTuning()
        assert t.method == "p1"
        assert t.unlock_k == 3
        assert t.profile == ExecutionProfile.GUARDED
        assert t.max_gap is None

    def test_extra_keys_rejected(self):
        with pytest.raises(ValidationError):
            CalibrateTuning(method="p1", bogus_knob=1.0)


# ----- Coercion helpers -----


class TestResolveProvider:
    def test_string_dispatches_to_factory(self, monkeypatch):
        captured: dict = {}

        def fake_make_provider(name, model=None, base_url=None):
            captured["name"] = name
            captured["model"] = model
            return "STUB_PROVIDER"

        monkeypatch.setattr("omegaprompt.runtime.make_provider", fake_make_provider)
        result = _resolve_provider("anthropic")
        assert result == "STUB_PROVIDER"
        assert captured["name"] == "anthropic"

    def test_provider_spec_dispatches(self, monkeypatch):
        captured: dict = {}

        def fake_make_provider(name, model=None, base_url=None):
            captured.update({"name": name, "model": model, "base_url": base_url})
            return "STUB"

        monkeypatch.setattr("omegaprompt.runtime.make_provider", fake_make_provider)
        _resolve_provider(ProviderSpec(name="openai", model="gpt-4o"))
        assert captured == {"name": "openai", "model": "gpt-4o", "base_url": None}

    def test_dict_dispatches(self, monkeypatch):
        captured: dict = {}

        def fake_make_provider(name, model=None, base_url=None):
            captured.update({"name": name, "model": model})
            return "STUB"

        monkeypatch.setattr("omegaprompt.runtime.make_provider", fake_make_provider)
        _resolve_provider({"name": "anthropic", "model": "claude-opus"})
        assert captured["name"] == "anthropic"
        assert captured["model"] == "claude-opus"


class TestResolveArtifact:
    def test_passthrough(self):
        a = _make_artifact()
        assert _resolve_artifact(a) is a

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            _resolve_artifact(42)


# ----- report() -----


class TestReport:
    def test_returns_markdown_string(self):
        artifact = _make_artifact()
        md = report(artifact)
        assert isinstance(md, str)
        assert len(md) > 0


# ----- diff() -----


class TestDiff:
    def test_no_regression_when_identical(self):
        a = _make_artifact()
        b = _make_artifact()
        result = diff(a, b)
        assert isinstance(result, ArtifactDiff)
        assert result.regressed is False
        assert result.fitness_delta == 0.0

    def test_regression_on_fitness_drop(self):
        old = _make_artifact(calibrated_fitness=0.80)
        new = _make_artifact(calibrated_fitness=0.65)
        result = diff(old, new)
        assert result.regressed is True
        assert result.fitness_delta == pytest.approx(-0.15)
        assert any("calibrated_fitness regressed" in r for r in result.regression_reasons)

    def test_regression_on_hard_gate_drop(self):
        old = _make_artifact(hard_gate_pass_rate=0.95)
        new = _make_artifact(hard_gate_pass_rate=0.80)
        result = diff(old, new)
        assert result.regressed is True
        assert any("hard_gate_pass_rate" in r for r in result.regression_reasons)

    def test_improvement_not_flagged(self):
        old = _make_artifact(calibrated_fitness=0.65)
        new = _make_artifact(calibrated_fitness=0.80)
        result = diff(old, new)
        assert result.regressed is False
        assert result.fitness_delta == pytest.approx(0.15)

    def test_markdown_format_returns_string(self):
        a = _make_artifact()
        md = diff(a, a, format="markdown")
        assert isinstance(md, str)
        assert "omegaprompt diff" in md
        assert "## OK" in md

    def test_markdown_shows_regression_section(self):
        old = _make_artifact(calibrated_fitness=0.80)
        new = _make_artifact(calibrated_fitness=0.65)
        md = diff(old, new, format="markdown")
        assert "## REGRESSION" in md
        assert "calibrated_fitness regressed" in md
