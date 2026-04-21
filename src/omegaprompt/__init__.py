"""omegaprompt - model-agnostic calibration discipline for LLM prompts.

v1.0 is a ground-up restructure of the v0.2 interface. The *discipline*
(sensitivity-driven coordinate descent, hard_gate x soft_score fitness,
walk-forward ship gate, machine-readable artifacts) is preserved intact;
the *contract* is provider-neutral: meta-axes instead of Claude-specific
knobs, a single ``call()`` method on :class:`LLMProvider`, a
:class:`Judge` protocol with rule / LLM / ensemble implementations.

Public API:

    from omegaprompt import (
        # domain
        Dataset, DatasetItem,
        PromptVariants, MetaAxisSpace, ResolvedPromptParams,
        JudgeRubric, Dimension, HardGate, JudgeResult,
        ReasoningProfile, OutputBudgetBucket,
        ResponseSchemaMode, ToolPolicyVariant,
        CalibrationArtifact, EvalResult, WalkForwardResult,
        # core
        CompositeFitness, aggregate_fitness, item_fitness,
        evaluate_walk_forward, measure_sensitivity,
        save_artifact, load_artifact,
        # providers
        LLMProvider, ProviderRequest, ProviderResponse,
        make_provider, supported_providers,
        # judges
        Judge, LLMJudge, RuleJudge, EnsembleJudge,
        # target
        PromptTarget,
    )

Depends on:
    omega-lock (>=0.1.4)  - CalibrableTarget host + search pipeline
    anthropic (>=0.40.0)
    openai (>=1.50.0)
"""

from omegaprompt.core import (
    CompositeFitness,
    aggregate_fitness,
    evaluate_walk_forward,
    item_fitness,
    load_artifact,
    measure_sensitivity,
    save_artifact,
    select_unlocked_axes,
)
from omegaprompt.domain import (
    CalibrationArtifact,
    Dataset,
    DatasetItem,
    Dimension,
    EvalItemResult,
    EvalResult,
    HardGate,
    HardGateFlags,
    JudgeResult,
    JudgeRubric,
    MetaAxisSpace,
    OutputBudgetBucket,
    PerItemScore,
    PromptVariants,
    ReasoningProfile,
    ResolvedPromptParams,
    ResponseSchemaMode,
    ToolPolicyVariant,
    WalkForwardResult,
)
from omegaprompt.judges import (
    EnsembleJudge,
    Judge,
    JudgeError,
    LLMJudge,
    RuleCheck,
    RuleJudge,
)
from omegaprompt.providers import (
    DEFAULT_MODELS,
    LLMProvider,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    make_provider,
    supported_providers,
)
from omegaprompt.targets import PromptTarget

__version__ = "1.0.0"

__all__ = [
    # domain
    "CalibrationArtifact",
    "Dataset",
    "DatasetItem",
    "Dimension",
    "EvalItemResult",
    "EvalResult",
    "HardGate",
    "HardGateFlags",
    "JudgeResult",
    "JudgeRubric",
    "MetaAxisSpace",
    "OutputBudgetBucket",
    "PerItemScore",
    "PromptVariants",
    "ReasoningProfile",
    "ResolvedPromptParams",
    "ResponseSchemaMode",
    "ToolPolicyVariant",
    "WalkForwardResult",
    # core
    "CompositeFitness",
    "aggregate_fitness",
    "item_fitness",
    "evaluate_walk_forward",
    "measure_sensitivity",
    "select_unlocked_axes",
    "save_artifact",
    "load_artifact",
    # providers
    "LLMProvider",
    "ProviderError",
    "ProviderRequest",
    "ProviderResponse",
    "DEFAULT_MODELS",
    "make_provider",
    "supported_providers",
    # judges
    "Judge",
    "JudgeError",
    "LLMJudge",
    "RuleCheck",
    "RuleJudge",
    "EnsembleJudge",
    # target
    "PromptTarget",
    # version
    "__version__",
]
