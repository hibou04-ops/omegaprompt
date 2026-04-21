"""Target adapters - CalibrableTarget implementations.

A target binds (provider, judge, dataset, rubric, variants, space) into a
single object with the ``param_space()`` / ``evaluate(params)`` interface
omega-lock's search layer expects. Each call to ``evaluate`` issues one
provider call per dataset item, plus one judge call per item.
"""

from omegaprompt.targets.base import CalibrableTarget
from omegaprompt.targets.prompt_target import PromptTarget

__all__ = ["CalibrableTarget", "PromptTarget"]
