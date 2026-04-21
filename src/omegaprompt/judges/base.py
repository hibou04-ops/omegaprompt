"""Judge Protocol - provider-neutral scoring contract.

Every judge consumes (rubric, dataset_item, target_response) and returns
a ``JudgeResult`` + usage dict. No judge implementation may import a
provider SDK directly - they go through :class:`LLMProvider` if they need
an LLM call.
"""

from __future__ import annotations

from typing import Protocol

from omegaprompt.domain.dataset import DatasetItem
from omegaprompt.domain.judge import JudgeResult, JudgeRubric


class JudgeError(RuntimeError):
    """Raised when a judge cannot produce a valid result."""


class Judge(Protocol):
    """Structural interface for judges."""

    name: str

    def score(
        self,
        *,
        rubric: JudgeRubric,
        item: DatasetItem,
        target_response: str,
    ) -> tuple[JudgeResult, dict[str, int]]:
        """Return ``(result, usage)``.

        ``usage`` uses the same canonical key set as
        :func:`omegaprompt.providers.base.empty_usage`; rule-based judges
        return a zero-valued dict.
        """
        ...
