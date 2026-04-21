"""Deterministic rule-based judge.

``RuleJudge`` evaluates ``hard_gates`` with ``evaluator='rule'`` against a
user-supplied set of :class:`RuleCheck` callables, plus optional default
checks for common patterns. It does not emit rubric dimension scores -
when a calibration's dimensions require qualitative judgment, combine
``RuleJudge`` with :class:`LLMJudge` via :class:`EnsembleJudge`.

Why this exists: many calibrations' hard gates (format validity, refusal
detection) are deterministic shape checks. Running them in Python rather
than a second LLM call is cheaper, reproducible, and removes a source of
judge drift. The LLM judge remains the authority on the soft score.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from omegaprompt.domain.dataset import DatasetItem
from omegaprompt.domain.judge import JudgeResult, JudgeRubric
from omegaprompt.judges.base import JudgeError
from omegaprompt.providers.base import empty_usage


# A rule check is a callable: (target_response, item) -> bool (True = pass).
RuleCheckFn = Callable[[str, DatasetItem], bool]


@dataclass(frozen=True)
class RuleCheck:
    """A single deterministic gate check."""

    name: str
    check: RuleCheckFn


_REFUSAL_PATTERNS = (
    r"\bI (can(?:not|'t)|won't)\b.*?\b(help|assist|provide|generate|write|do)\b",
    r"\bI('m| am) (unable|not able)\b.*?\b(help|assist|provide|do)\b",
    r"\bI (refuse|must decline|cannot comply)\b",
    r"^\s*(sorry|apologies)[,.].*?\b(cannot|can't|won't|unable)\b",
)


def default_no_refusal() -> RuleCheck:
    """Fail when the response matches common refusal phrasings."""
    compiled = [re.compile(p, re.IGNORECASE) for p in _REFUSAL_PATTERNS]

    def _check(response: str, _item: DatasetItem) -> bool:
        return not any(p.search(response) for p in compiled)

    return RuleCheck(name="no_refusal", check=_check)


def default_non_empty() -> RuleCheck:
    def _check(response: str, _item: DatasetItem) -> bool:
        return bool(response and response.strip())

    return RuleCheck(name="non_empty", check=_check)


def json_object_check(name: str = "format_valid") -> RuleCheck:
    """Fail when the response is not a parseable JSON object.

    Tolerates leading/trailing whitespace and optional markdown fences.
    """

    def _check(response: str, _item: DatasetItem) -> bool:
        text = response.strip()
        if text.startswith("```"):
            # strip first fence line and last fence line
            lines = text.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                text = "\n".join(lines[1:-1])
        try:
            parsed = json.loads(text)
        except Exception:
            return False
        return isinstance(parsed, dict)

    return RuleCheck(name=name, check=_check)


def regex_check(name: str, pattern: str, flags: int = 0) -> RuleCheck:
    compiled = re.compile(pattern, flags)

    def _check(response: str, _item: DatasetItem) -> bool:
        return bool(compiled.search(response))

    return RuleCheck(name=name, check=_check)


class RuleJudge:
    """Deterministic judge - evaluates rule-mode hard gates only.

    ``rubric.dimensions`` are NOT scored by RuleJudge; any dimension score
    the caller needs must come from a different judge (typically LLMJudge
    via EnsembleJudge).

    Construction:

    .. code-block:: python

        judge = RuleJudge(
            checks=[
                default_no_refusal(),
                default_non_empty(),
                json_object_check(name="format_valid"),
            ]
        )
    """

    name = "rule"

    def __init__(self, *, checks: list[RuleCheck]) -> None:
        if not checks:
            raise JudgeError("RuleJudge requires at least one check.")
        names = [c.name for c in checks]
        if len(set(names)) != len(names):
            raise JudgeError(f"RuleCheck names must be unique; got {names!r}.")
        self.checks = {c.name: c for c in checks}

    def score(
        self,
        *,
        rubric: JudgeRubric,
        item: DatasetItem,
        target_response: str,
    ) -> tuple[JudgeResult, dict[str, int]]:
        gate_results: dict[str, bool] = {}
        for gate in rubric.hard_gates:
            if gate.evaluator != "rule":
                continue
            if gate.name not in self.checks:
                raise JudgeError(
                    f"Gate {gate.name!r} declared evaluator='rule' but no RuleCheck "
                    "with that name was registered with this RuleJudge."
                )
            check = self.checks[gate.name]
            gate_results[gate.name] = bool(check.check(target_response, item))

        # Dimensions are left unset (RuleJudge does not score them); callers
        # that need both dimensions and rule gates should use EnsembleJudge.
        zero_scores: dict[str, int] = {d.name: d.scale[0] for d in rubric.dimensions}
        notes = "rule-only judge; dimensions unscored"
        return (
            JudgeResult(scores=zero_scores, gate_results=gate_results, notes=notes),
            empty_usage(),
        )

    # Helper for EnsembleJudge: returns only the gate dict without manufacturing
    # placeholder dimension scores.
    def evaluate_gates(
        self, rubric: JudgeRubric, item: DatasetItem, target_response: str
    ) -> dict[str, bool]:
        gate_results: dict[str, bool] = {}
        for gate in rubric.hard_gates:
            if gate.evaluator != "rule":
                continue
            if gate.name not in self.checks:
                raise JudgeError(
                    f"Gate {gate.name!r} declared evaluator='rule' but no RuleCheck "
                    "with that name was registered."
                )
            gate_results[gate.name] = bool(self.checks[gate.name].check(target_response, item))
        return gate_results


__all__ = [
    "RuleCheck",
    "RuleCheckFn",
    "RuleJudge",
    "default_no_refusal",
    "default_non_empty",
    "json_object_check",
    "regex_check",
]
