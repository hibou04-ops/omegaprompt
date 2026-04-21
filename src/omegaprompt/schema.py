"""Backward-compat shim - re-exports from the v1.0 domain layer.

The v0.2 ``schema.py`` co-located ``ParamVariants``, ``PromptSpace``, and
``CalibrationOutcome``. In v1.0 they've split:

- ``ParamVariants`` -> ``omegaprompt.domain.params.PromptVariants``
- ``PromptSpace`` -> ``omegaprompt.domain.params.MetaAxisSpace`` (with
  Claude-specific axes replaced by provider-neutral meta-axes).
- ``CalibrationOutcome`` -> ``omegaprompt.domain.result.CalibrationArtifact``
  (richer schema: sensitivity_ranking, walk_forward block, status).

The legacy names are aliased below for import-path compatibility. The
underlying shape changed in v1.0 though - any code that constructed a
``PromptSpace`` by name will need migration.
"""

from omegaprompt.domain.params import MetaAxisSpace, PromptVariants
from omegaprompt.domain.result import CalibrationArtifact

# Legacy names kept for import-path continuity. Functional equivalents are
# not guaranteed; see CHANGELOG for the v0.2 -> v1.0 migration guide.
ParamVariants = PromptVariants
PromptSpace = MetaAxisSpace
CalibrationOutcome = CalibrationArtifact

__all__ = [
    "ParamVariants",
    "PromptSpace",
    "CalibrationOutcome",
    "PromptVariants",
    "MetaAxisSpace",
    "CalibrationArtifact",
]
