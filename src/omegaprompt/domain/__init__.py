"""Provider-neutral domain contracts for omegaprompt v1.0.

The calibration discipline (sensitivity, walk-forward, fitness, artifacts)
is vendor-agnostic. Everything a provider adapter or target/judge
implementation consumes lives in this package, so the core kernel has a
single boundary it depends on.

Key contract changes from v0.2.x:

- Claude-specific axes (``effort_idx``, ``thinking_enabled``,
  ``max_tokens_bucket``) are replaced by provider-neutral meta-axes
  (``reasoning_profile``, ``output_budget_bucket``, ``response_schema_mode``,
  ``tool_policy_variant``). Each provider adapter maps these to its
  vendor's native parameters.
- ``MetaAxisSpace`` replaces ``PromptSpace``. The bounds are declared in
  terms of enums, not integer indices.
"""

from omegaprompt.domain.dataset import Dataset, DatasetItem
from omegaprompt.domain.enums import (
    OutputBudgetBucket,
    ReasoningProfile,
    ResponseSchemaMode,
    ToolPolicyVariant,
)
from omegaprompt.domain.judge import (
    Dimension,
    HardGate,
    HardGateFlags,
    JudgeResult,
    JudgeRubric,
)
from omegaprompt.domain.params import (
    MetaAxisSpace,
    PromptVariants,
    ResolvedPromptParams,
)
from omegaprompt.domain.result import (
    CalibrationArtifact,
    EvalItemResult,
    EvalResult,
    PerItemScore,
    WalkForwardResult,
)

__all__ = [
    "Dataset",
    "DatasetItem",
    "OutputBudgetBucket",
    "ReasoningProfile",
    "ResponseSchemaMode",
    "ToolPolicyVariant",
    "Dimension",
    "HardGate",
    "HardGateFlags",
    "JudgeResult",
    "JudgeRubric",
    "MetaAxisSpace",
    "PromptVariants",
    "ResolvedPromptParams",
    "CalibrationArtifact",
    "EvalItemResult",
    "EvalResult",
    "PerItemScore",
    "WalkForwardResult",
]
