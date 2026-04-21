"""Composite fitness: ``hard_gate × soft_score``.

Contract preserved from v0.2: if any hard gate fails on a dataset item,
that item's contribution is zero. This is the structural defense against
prompts that score beautifully on the soft rubric but refuse on a subset
of inputs or emit malformed output.

v1.0 change: exposes standalone ``item_fitness`` / ``aggregate_fitness``
functions alongside the ``CompositeFitness`` stateful aggregator, so
rule/ensemble judges can compose fitness without going through a class
instance.
"""

from __future__ import annotations

from collections.abc import Iterable

from omegaprompt.domain.judge import JudgeResult, JudgeRubric
from omegaprompt.domain.result import PerItemScore


def item_fitness(judge: JudgeResult, rubric: JudgeRubric) -> float:
    """Single-item fitness: 0 if any gate failed, else weighted soft score."""
    if judge.any_gate_failed():
        return 0.0
    return judge.weighted_score(rubric)


def aggregate_fitness(items: Iterable[PerItemScore]) -> float:
    """Mean final_score across items. Empty input -> 0.0."""
    items = list(items)
    if not items:
        return 0.0
    return sum(p.final_score for p in items) / len(items)


class CompositeFitness:
    """Stateful per-run aggregator.

    Holds the last batch's per-item breakdown for reporting. ``evaluate``
    is called once per ``CalibrableTarget.evaluate()``; ``pass_rate`` is
    queried after.
    """

    def __init__(self, rubric: JudgeRubric) -> None:
        self.rubric = rubric
        self.last_per_item: list[PerItemScore] = []

    def evaluate(self, judge_results: Iterable[tuple[str, JudgeResult]]) -> float:
        per_item: list[PerItemScore] = []
        for item_id, jr in judge_results:
            soft = jr.weighted_score(self.rubric)
            passed = not jr.any_gate_failed()
            per_item.append(
                PerItemScore(
                    item_id=item_id,
                    soft_score=soft,
                    gates_passed=passed,
                    final_score=soft if passed else 0.0,
                    notes=jr.notes,
                )
            )
        self.last_per_item = per_item
        return aggregate_fitness(per_item)

    def pass_rate(self) -> float:
        if not self.last_per_item:
            return 0.0
        return sum(1 for p in self.last_per_item if p.gates_passed) / len(self.last_per_item)
