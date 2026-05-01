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
from omegaprompt.judges.base import Judge, JudgeError, JudgeOutcome
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
    ) -> JudgeOutcome:
        rule_gates = self.rule_judge.evaluate_gates(rubric, item, target_response)
        any_rule_failed = any(result is False for result in rule_gates.values())

        usage = empty_usage()

        if any_rule_failed:
            # Short-circuit: rule gate failed, fitness will collapse anyway.
            zero_scores = {d.name: d.scale[0] for d in rubric.dimensions}
            return JudgeOutcome(
                result=JudgeResult(
                    scores=zero_scores,
                    gate_results=rule_gates,
                    notes="rule gate failed; LLM-judge skipped",
                ),
                usage=usage,
                degraded_capabilities=[],
                latency_ms=0.0,
            )

        # Escalate to LLM judge for dimensions + judge-mode gates. The
        # fallback may itself be a JudgeOutcome-returning judge (LLMJudge
        # post P0) or a tuple-returning legacy judge — JudgeOutcome's
        # __iter__ handles both via the same destructuring statement.
        fallback_outcome = self.fallback.score(
            rubric=rubric, item=item, target_response=target_response
        )
        fallback_result, fallback_usage = fallback_outcome
        usage = _accumulate_usage(usage, fallback_usage)

        merged_gates = {**rule_gates, **fallback_result.gate_results}
        merged = JudgeResult(
            scores=fallback_result.scores,
            gate_results=merged_gates,
            notes=fallback_result.notes,
        )
        # Propagate fallback's degradation/latency when available
        # (LLMJudge attaches them); legacy tuple-returning judges
        # contribute empty defaults via JudgeOutcome's defaults.
        return JudgeOutcome(
            result=merged,
            usage=usage,
            degraded_capabilities=list(
                getattr(fallback_outcome, "degraded_capabilities", []) or []
            ),
            latency_ms=float(getattr(fallback_outcome, "latency_ms", 0.0) or 0.0),
        )


__all__ = ["EnsembleJudge"]
