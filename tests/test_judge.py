"""Judge rubric + result tests."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from omegaprompt.judge import Dimension, HardGate, JudgeResult, JudgeRubric
from tests.helpers import workspace_tmpdir


def _rubric() -> JudgeRubric:
    return JudgeRubric(
        dimensions=[
            Dimension(name="correctness", description="is it right", weight=0.5),
            Dimension(name="clarity", description="is it clear", weight=0.3),
            Dimension(name="conciseness", description="not padded", weight=0.2),
        ],
        hard_gates=[
            HardGate(name="no_refusal", description="did it attempt"),
        ],
    )


def test_dimension_scale_valid():
    Dimension(name="x", description="x", weight=1.0, scale=(1, 5))


def test_dimension_scale_invalid():
    with pytest.raises(ValidationError):
        Dimension(name="x", description="x", weight=1.0, scale=(5, 1))


def test_rubric_requires_at_least_one_dimension():
    with pytest.raises(ValidationError):
        JudgeRubric(dimensions=[])


def test_rubric_rejects_duplicate_dimension_names():
    with pytest.raises(ValidationError):
        JudgeRubric(
            dimensions=[
                Dimension(name="x", description="a", weight=1.0),
                Dimension(name="x", description="b", weight=1.0),
            ]
        )


def test_rubric_rejects_zero_total_weight():
    with pytest.raises(ValidationError):
        JudgeRubric(
            dimensions=[
                Dimension(name="a", description="a", weight=0.0),
                Dimension(name="b", description="b", weight=0.0),
            ]
        )


def test_rubric_normalized_weights_sum_to_one():
    r = _rubric()
    weights = r.normalized_weights()
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    # ratios preserved
    assert abs(weights["correctness"] - 0.5) < 1e-9


def test_rubric_from_json():
    with workspace_tmpdir() as tmp_path:
        p = tmp_path / "rubric.json"
        p.write_text(
            json.dumps(
                {
                    "dimensions": [
                        {
                            "name": "c",
                            "description": "c",
                            "weight": 1.0,
                            "scale": [1, 5],
                        }
                    ],
                    "hard_gates": [
                        {"name": "g", "description": "g", "evaluator": "judge"}
                    ],
                }
            ),
            encoding="utf-8",
        )
        r = JudgeRubric.from_json(p)
        assert r.dimensions[0].name == "c"
        assert r.hard_gates[0].name == "g"


def test_judge_result_weighted_score():
    r = _rubric()
    jr = JudgeResult(
        scores={"correctness": 5, "clarity": 3, "conciseness": 4},
        gate_results={"no_refusal": True},
    )
    # normalized: corr=1.0, clarity=0.5, conciseness=0.75
    # weighted: 0.5*1.0 + 0.3*0.5 + 0.2*0.75 = 0.5 + 0.15 + 0.15 = 0.8
    assert abs(jr.weighted_score(r) - 0.8) < 1e-9


def test_judge_result_clamps_out_of_scale():
    r = _rubric()
    jr = JudgeResult(
        scores={"correctness": 99, "clarity": 0, "conciseness": 3},
    )
    # corr clamped to 5, clarity clamped to 1
    # normalized: corr=1.0, clarity=0.0, conciseness=0.5
    # weighted: 0.5 + 0 + 0.1 = 0.6
    assert abs(jr.weighted_score(r) - 0.6) < 1e-9


def test_judge_result_any_gate_failed():
    jr = JudgeResult(
        scores={"correctness": 3, "clarity": 3, "conciseness": 3},
        gate_results={"a": True, "b": False},
    )
    assert jr.any_gate_failed() is True


def test_judge_result_no_gates_means_no_failure():
    jr = JudgeResult(scores={"correctness": 3, "clarity": 3, "conciseness": 3})
    assert jr.any_gate_failed() is False


def test_judge_result_ignores_unknown_dimensions():
    r = _rubric()
    jr = JudgeResult(scores={"correctness": 5, "unknown_dim": 3})
    # only correctness counts: 0.5 * 1.0 = 0.5
    assert abs(jr.weighted_score(r) - 0.5) < 1e-9
