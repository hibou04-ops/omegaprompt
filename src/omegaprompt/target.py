"""Backward-compat shim - re-exports from :mod:`omegaprompt.targets`.

Prefer ``from omegaprompt.targets import PromptTarget`` in new code.
The v1.0 ``PromptTarget`` signature differs from v0.2: it takes a
``Judge`` object (not a separate ``judge_provider`` + hard-coded judge
call) and a ``PromptVariants`` / ``MetaAxisSpace`` instead of the old
``ParamVariants`` / ``PromptSpace``.
"""

from omegaprompt.targets.prompt_target import PromptTarget

__all__ = ["PromptTarget"]
