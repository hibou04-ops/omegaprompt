"""CompositeFitness aggregation tests."""

from omegaprompt.fitness import CompositeFitness
from omegaprompt.judge import Dimension, HardGate, JudgeResult, JudgeRubric


def _rubric() -> JudgeRubric:
    return JudgeRubric(
        dimensions=[
            Dimension(name="a", description="a", weight=1.0, scale=(0, 1)),
        ],
        hard_gates=[HardGate(name="g", description="g")],
    )


def test_empty_batch_yields_zero():
    cf = CompositeFitness(_rubric())
    assert cf.evaluate([]) == 0.0
    assert cf.pass_rate() == 0.0


def test_all_pass_gates_weighted_mean():
    cf = CompositeFitness(_rubric())
    results = [
        ("t1", JudgeResult(scores={"a": 1}, gate_results={"g": True})),
        ("t2", JudgeResult(scores={"a": 0}, gate_results={"g": True})),
    ]
    score = cf.evaluate(results)
    # normalized: 1.0, 0.0 -> mean 0.5
    assert abs(score - 0.5) < 1e-9
    assert cf.pass_rate() == 1.0


def test_gate_failure_zeroes_item():
    cf = CompositeFitness(_rubric())
    results = [
        ("t1", JudgeResult(scores={"a": 1}, gate_results={"g": True})),
        ("t2", JudgeResult(scores={"a": 1}, gate_results={"g": False})),
    ]
    score = cf.evaluate(results)
    # t1=1.0, t2=0 (gate failed) -> mean 0.5
    assert abs(score - 0.5) < 1e-9
    assert cf.pass_rate() == 0.5


def test_all_fail_gates_collapses_to_zero():
    cf = CompositeFitness(_rubric())
    results = [
        ("t1", JudgeResult(scores={"a": 1}, gate_results={"g": False})),
        ("t2", JudgeResult(scores={"a": 1}, gate_results={"g": False})),
    ]
    score = cf.evaluate(results)
    assert score == 0.0
    assert cf.pass_rate() == 0.0


def test_per_item_breakdown_preserved():
    cf = CompositeFitness(_rubric())
    results = [
        ("t1", JudgeResult(scores={"a": 1}, gate_results={"g": True}, notes="ok")),
        ("t2", JudgeResult(scores={"a": 0}, gate_results={"g": False}, notes="refused")),
    ]
    cf.evaluate(results)
    assert len(cf.last_per_item) == 2
    assert cf.last_per_item[0].item_id == "t1"
    assert cf.last_per_item[0].gates_passed is True
    assert cf.last_per_item[1].gates_passed is False
    assert cf.last_per_item[1].final_score == 0.0
    assert cf.last_per_item[1].notes == "refused"
