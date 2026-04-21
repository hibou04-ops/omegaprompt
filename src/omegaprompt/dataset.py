"""Backward-compat shim - re-exports from :mod:`omegaprompt.domain.dataset`.

Kept so v0.2 import paths (``from omegaprompt.dataset import Dataset``)
keep working during the v1.0 transition. Prefer ``from omegaprompt.domain
import Dataset, DatasetItem`` in new code.
"""

from omegaprompt.domain.dataset import Dataset, DatasetItem

__all__ = ["Dataset", "DatasetItem"]
