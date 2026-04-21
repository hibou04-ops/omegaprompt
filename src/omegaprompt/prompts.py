"""Backward-compat shim - re-exports :data:`JUDGE_SYSTEM_PROMPT`.

The canonical location in v1.0 is :mod:`omegaprompt.judges.llm_judge`.
"""

from omegaprompt.judges.llm_judge import JUDGE_SYSTEM_PROMPT

__all__ = ["JUDGE_SYSTEM_PROMPT"]
