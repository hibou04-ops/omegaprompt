"""Backward-compat shim - re-exports from :mod:`omegaprompt.core.fitness`.

Prefer ``from omegaprompt.core import CompositeFitness, aggregate_fitness,
item_fitness`` in new code.
"""

from omegaprompt.core.fitness import CompositeFitness, aggregate_fitness, item_fitness
from omegaprompt.domain.result import PerItemScore

__all__ = ["CompositeFitness", "PerItemScore", "aggregate_fitness", "item_fitness"]
