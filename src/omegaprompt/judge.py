"""Backward-compat shim - re-exports from the v1.0 domain/judges layout.

- Rubric types (Dimension, HardGate, JudgeRubric, JudgeResult) live in
  :mod:`omegaprompt.domain.judge`.
- Judge implementations (LLMJudge, RuleJudge, EnsembleJudge) live in
  :mod:`omegaprompt.judges`.

Prefer the new import paths in new code.
"""

from omegaprompt.domain.judge import (
    Dimension,
    HardGate,
    HardGateFlags,
    JudgeResult,
    JudgeRubric,
)
from omegaprompt.judges import EnsembleJudge, Judge, JudgeError, LLMJudge, RuleCheck, RuleJudge

__all__ = [
    "Dimension",
    "HardGate",
    "HardGateFlags",
    "JudgeResult",
    "JudgeRubric",
    "EnsembleJudge",
    "Judge",
    "JudgeError",
    "LLMJudge",
    "RuleCheck",
    "RuleJudge",
]
