"""Ensemble judge: rule gates first, LLM-judge second.

The ensemble pattern gives you the rule judge's cost savings on clearly
broken responses (malformed format, flat refusal) and the LLM judge's
qualitative scoring on responses that pass the structural bar. Items
failing any rule gate short-circuit the LLM call entirely.

The returned :class:`JudgeResult` merges gate results from both sides:
rule-mode gates come from the :class:`RuleJudge`, judge-mode gates come
from the :class:`LLMJudge`. Dimension scores always come from the LLM.

When the rule judge fails a gate, the LLM call is skipped; dimensions are
filled with their scale minimum (since the item's final_score will be 0
anyway via the hard-gate collapse). This is a cost optimization, not a
scoring change.
"""

from __future__ import annotations

from omegaprompt.domain.dataset import DatasetItem
from omegaprompt.domain.judge import JudgeResult, JudgeRubric
from omegaprompt.judges.base import Judge, JudgeError
from omegaprompt.judges.rule_judge import RuleJudge
from omegaprompt.providers.base import empty_usage


def _accumulate_usage(acc: dict[str, int], delta: dict[str, int]) -> dict[str, int]:
    out = dict(acc)
    for k, v in delta.items():
        out[k] = out.get(k, 0) + int(v or 0)
    return out


class EnsembleJudge:
    """Run ``RuleJudge`` first; escalate to ``LLMJudge`` only on pass."""

    name = "ensemble"

    def __init__(self, *, rule_judge: RuleJudge, fallback: Judge) -> None:
        if not isinstance(rule_judge, RuleJudge):
            raise JudgeError("EnsembleJudge.rule_judge must be a RuleJudge instance.")
        self.rule_judge = rule_judge
        self.fallback = fallback

    def score(
        self,
        *,
        rubric: JudgeRubric,
        item: DatasetItem,
        target_response: str,
    ) -> tuple[JudgeResult, dict[str, int]]:
        rule_gates = self.rule_judge.evaluate_gates(rubric, item, target_response)
        any_rule_failed = any(result is False for result in rule_gates.values())

        usage = empty_usage()

        if any_rule_failed:
            # Short-circuit: rule gate failed, fitness will collapse anyway.
            zero_scores = {d.name: d.scale[0] for d in rubric.dimensions}
            return (
                JudgeResult(
                    scores=zero_scores,
                    gate_results=rule_gates,
                    notes="rule gate failed; LLM-judge skipped",
                ),
                usage,
            )

        # Escalate to LLM judge for dimensions + judge-mode gates.
        fallback_result, fallback_usage = self.fallback.score(
            rubric=rubric, item=item, target_response=target_response
        )
        usage = _accumulate_usage(usage, fallback_usage)

        merged_gates = {**rule_gates, **fallback_result.gate_results}
        merged = JudgeResult(
            scores=fallback_result.scores,
            gate_results=merged_gates,
            notes=fallback_result.notes,
        )
        return merged, usage


__all__ = ["EnsembleJudge"]
