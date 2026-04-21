"""Judge implementations.

Three concrete judges ship in v1.0:

- :class:`LLMJudge` - provider-neutral rubric-based LLM grader. The
  default for most calibrations.
- :class:`RuleJudge` - deterministic gate checker (regex / JSON schema /
  exact match / custom callables). Cheap, reproducible, no API cost.
  Useful when a calibration's hard gates are all structural.
- :class:`EnsembleJudge` - runs a ``RuleJudge`` first and escalates to a
  fallback judge only when rule gates pass. Cuts LLM-judge cost on
  obviously-broken responses.
"""

from omegaprompt.judges.base import Judge, JudgeError
from omegaprompt.judges.ensemble_judge import EnsembleJudge
from omegaprompt.judges.llm_judge import LLMJudge
from omegaprompt.judges.rule_judge import RuleCheck, RuleJudge

__all__ = [
    "Judge",
    "JudgeError",
    "LLMJudge",
    "RuleJudge",
    "RuleCheck",
    "EnsembleJudge",
]
