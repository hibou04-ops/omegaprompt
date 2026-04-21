"""Tests for the Judge implementations (rule, LLM, ensemble)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from omegaprompt.domain.dataset import DatasetItem
from omegaprompt.domain.judge import Dimension, HardGate, JudgeResult, JudgeRubric
from omegaprompt.judges import EnsembleJudge, JudgeError, LLMJudge, RuleJudge
from omegaprompt.judges.rule_judge import (
    RuleCheck,
    default_no_refusal,
    default_non_empty,
    json_object_check,
    regex_check,
)
from omegaprompt.providers.base import ProviderResponse


def _item(id_: str = "t1", input_: str = "in", reference: str | None = None) -> DatasetItem:
    return DatasetItem(id=id_, input=input_, reference=reference)


def _rubric_rule_only() -> JudgeRubric:
    return JudgeRubric(
        dimensions=[Dimension(name="q", description="q", weight=1.0, scale=(0, 1))],
        hard_gates=[
            HardGate(name="no_refusal", description="r", evaluator="rule"),
            HardGate(name="format_valid", description="fmt", evaluator="rule"),
        ],
    )


def _rubric_judge_gates() -> JudgeRubric:
    return JudgeRubric(
        dimensions=[Dimension(name="q", description="q", weight=1.0)],
        hard_gates=[
            HardGate(name="no_refusal", description="r", evaluator="rule"),
            HardGate(name="correctness", description="c", evaluator="judge"),
        ],
    )


# --------------------------- RuleJudge ---------------------------


def test_rule_judge_requires_checks():
    with pytest.raises(JudgeError, match="at least one check"):
        RuleJudge(checks=[])


def test_rule_judge_rejects_duplicate_check_names():
    with pytest.raises(JudgeError, match="unique"):
        RuleJudge(
            checks=[
                RuleCheck(name="dup", check=lambda r, i: True),
                RuleCheck(name="dup", check=lambda r, i: False),
            ]
        )


def test_rule_judge_scores_rule_gates_only():
    judge = RuleJudge(
        checks=[
            default_no_refusal(),
            json_object_check(name="format_valid"),
        ]
    )
    result, usage = judge.score(
        rubric=_rubric_rule_only(),
        item=_item(),
        target_response='{"ok": true}',
    )
    assert result.gate_results["no_refusal"] is True
    assert result.gate_results["format_valid"] is True
    assert usage["input_tokens"] == 0


def test_rule_judge_detects_refusal_pattern():
    judge = RuleJudge(checks=[default_no_refusal()])
    result, _ = judge.score(
        rubric=JudgeRubric(
            dimensions=[Dimension(name="q", description="q", weight=1.0)],
            hard_gates=[HardGate(name="no_refusal", description="r", evaluator="rule")],
        ),
        item=_item(),
        target_response="I cannot help with that.",
    )
    assert result.gate_results["no_refusal"] is False


def test_rule_judge_json_check_handles_markdown_fence():
    judge = RuleJudge(checks=[json_object_check(name="format_valid")])
    rubric = JudgeRubric(
        dimensions=[Dimension(name="q", description="q", weight=1.0)],
        hard_gates=[HardGate(name="format_valid", description="f", evaluator="rule")],
    )
    result, _ = judge.score(
        rubric=rubric,
        item=_item(),
        target_response='```json\n{"x":1}\n```',
    )
    assert result.gate_results["format_valid"] is True


def test_rule_judge_regex_check_matches():
    rubric = JudgeRubric(
        dimensions=[Dimension(name="q", description="q", weight=1.0)],
        hard_gates=[HardGate(name="has_answer", description="x", evaluator="rule")],
    )
    judge = RuleJudge(checks=[regex_check("has_answer", r"ANSWER:\s*\S+")])
    result, _ = judge.score(rubric=rubric, item=_item(), target_response="ANSWER: 42")
    assert result.gate_results["has_answer"] is True


def test_rule_judge_non_empty_catches_blank():
    rubric = JudgeRubric(
        dimensions=[Dimension(name="q", description="q", weight=1.0)],
        hard_gates=[HardGate(name="non_empty", description="x", evaluator="rule")],
    )
    judge = RuleJudge(checks=[default_non_empty()])
    result, _ = judge.score(rubric=rubric, item=_item(), target_response="   ")
    assert result.gate_results["non_empty"] is False


def test_rule_judge_missing_check_raises():
    rubric = JudgeRubric(
        dimensions=[Dimension(name="q", description="q", weight=1.0)],
        hard_gates=[HardGate(name="missing", description="x", evaluator="rule")],
    )
    judge = RuleJudge(checks=[default_non_empty()])
    with pytest.raises(JudgeError, match="declared evaluator='rule'"):
        judge.score(rubric=rubric, item=_item(), target_response="x")


# --------------------------- LLMJudge ---------------------------


def test_llm_judge_calls_provider_with_strict_schema():
    provider = MagicMock()
    expected = JudgeResult(
        scores={"q": 4},
        gate_results={"correctness": True},
        notes="ok",
    )
    provider.call.return_value = ProviderResponse(
        parsed=expected,
        usage={
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 4096,
        },
    )
    judge = LLMJudge(provider=provider)
    result, usage = judge.score(
        rubric=_rubric_judge_gates(),
        item=_item(reference="ref"),
        target_response="target output",
    )
    assert result == expected
    assert usage["cache_read_input_tokens"] == 4096
    request = provider.call.call_args.args[0]
    from omegaprompt.domain.enums import ResponseSchemaMode

    assert request.response_schema_mode == ResponseSchemaMode.STRICT_SCHEMA
    assert request.output_schema is JudgeResult
    # Payload includes rubric + input + reference + response
    assert "<rubric>" in request.user_message
    assert "<reference>" in request.user_message
    assert "<response>" in request.user_message


def test_llm_judge_omits_reference_block_when_absent():
    provider = MagicMock()
    provider.call.return_value = ProviderResponse(
        parsed=JudgeResult(scores={"q": 3}),
        usage={},
    )
    judge = LLMJudge(provider=provider)
    judge.score(
        rubric=_rubric_judge_gates(),
        item=_item(reference=None),
        target_response="out",
    )
    payload = provider.call.call_args.args[0].user_message
    assert "<reference>" not in payload


def test_llm_judge_raises_when_provider_returns_non_judgeresult():
    provider = MagicMock()
    provider.call.return_value = ProviderResponse(parsed=None)
    judge = LLMJudge(provider=provider)
    with pytest.raises(JudgeError, match="did not return a JudgeResult"):
        judge.score(
            rubric=_rubric_judge_gates(),
            item=_item(),
            target_response="x",
        )


# --------------------------- EnsembleJudge ---------------------------


def test_ensemble_short_circuits_on_rule_failure():
    rule = RuleJudge(checks=[default_no_refusal()])
    fallback = MagicMock()
    fallback.name = "llm"
    ensemble = EnsembleJudge(rule_judge=rule, fallback=fallback)

    result, usage = ensemble.score(
        rubric=_rubric_judge_gates(),
        item=_item(),
        target_response="I cannot help with that.",
    )
    assert result.gate_results["no_refusal"] is False
    fallback.score.assert_not_called()
    assert usage["input_tokens"] == 0


def test_ensemble_escalates_when_rules_pass():
    rule = RuleJudge(checks=[default_no_refusal()])
    fallback = MagicMock()
    fallback.name = "llm"
    fallback.score.return_value = (
        JudgeResult(
            scores={"q": 5},
            gate_results={"correctness": True},
            notes="llm notes",
        ),
        {"input_tokens": 100, "output_tokens": 40, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    )
    ensemble = EnsembleJudge(rule_judge=rule, fallback=fallback)

    result, usage = ensemble.score(
        rubric=_rubric_judge_gates(),
        item=_item(),
        target_response="A clean attempt at the task.",
    )
    assert result.gate_results["no_refusal"] is True
    assert result.gate_results["correctness"] is True
    assert result.scores["q"] == 5
    assert result.notes == "llm notes"
    assert usage["input_tokens"] == 100
    fallback.score.assert_called_once()


def test_ensemble_rejects_non_rule_judge():
    fake_rule = MagicMock()
    fallback = MagicMock()
    with pytest.raises(JudgeError, match="RuleJudge instance"):
        EnsembleJudge(rule_judge=fake_rule, fallback=fallback)
