"""omegacal / omegaprompt - provider-neutral prompt calibration engine.

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
        ExecutionProfile, ShipRecommendation, BoundaryWarning,
        # core
        CompositeFitness, aggregate_fitness, item_fitness,
        evaluate_walk_forward, measure_sensitivity, policy_for, assess_run_risk,
        save_artifact, load_artifact,
        # providers
        LLMProvider, ProviderRequest, ProviderResponse, ProviderCapabilities,
        make_provider, supported_providers,
        # judges
        Judge, LLMJudge, RuleJudge, EnsembleJudge,
        # target
        CalibrableTarget, PromptTarget,
    )

Depends on:
    omega-lock (>=0.1.4)  - CalibrableTarget host + search pipeline
    anthropic (>=0.40.0)
    openai (>=1.50.0)
"""

from omegaprompt.core import (
    CompositeFitness,
    assess_run_risk,
    aggregate_fitness,
    evaluate_walk_forward,
    item_fitness,
    load_artifact,
    measure_sensitivity,
    policy_for,
    relaxed_safeguards_for,
    save_artifact,
    select_unlocked_axes,
)
from omegaprompt.domain import (
    BoundaryWarning,
    CalibrationArtifact,
    Dataset,
    DatasetItem,
    Dimension,
    ExecutionProfile,
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
    RelaxedSafeguard,
    ReasoningProfile,
    ResolvedPromptParams,
    RiskCategory,
    ResponseSchemaMode,
    ShipRecommendation,
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
    CapabilityEvent,
    CapabilityTier,
    DEFAULT_MODELS,
    LLMProvider,
    ProviderCapabilities,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    estimate_cost_units,
    make_provider,
    quality_per_cost,
    quality_per_latency,
    supported_providers,
)
from omegaprompt.preflight import (
    AdaptationPlan,
    AnalyticalFinding,
    ParameterOverride,
    PreflightReport,
    PreflightSeverity,
    PreflightStatus,
    analytical_preflight,
    apply_adaptation_plan,
    derive_adaptation_plan,
    empirical_preflight,
)
from omegaprompt.targets import CalibrableTarget, PromptTarget

__version__ = "1.0.0"

__all__ = [
    # domain
    "CalibrationArtifact",
    "Dataset",
    "DatasetItem",
    "Dimension",
    "BoundaryWarning",
    "ExecutionProfile",
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
    "RelaxedSafeguard",
    "ReasoningProfile",
    "ResolvedPromptParams",
    "RiskCategory",
    "ResponseSchemaMode",
    "ShipRecommendation",
    "ToolPolicyVariant",
    "WalkForwardResult",
    # core
    "CompositeFitness",
    "assess_run_risk",
    "aggregate_fitness",
    "item_fitness",
    "evaluate_walk_forward",
    "measure_sensitivity",
    "policy_for",
    "relaxed_safeguards_for",
    "select_unlocked_axes",
    "save_artifact",
    "load_artifact",
    # providers
    "LLMProvider",
    "CapabilityEvent",
    "CapabilityTier",
    "ProviderError",
    "ProviderCapabilities",
    "ProviderRequest",
    "ProviderResponse",
    "DEFAULT_MODELS",
    "estimate_cost_units",
    "make_provider",
    "quality_per_cost",
    "quality_per_latency",
    "supported_providers",
    # judges
    "Judge",
    "JudgeError",
    "LLMJudge",
    "RuleCheck",
    "RuleJudge",
    "EnsembleJudge",
    # target
    "CalibrableTarget",
    "PromptTarget",
    # preflight
    "AdaptationPlan",
    "AnalyticalFinding",
    "ParameterOverride",
    "PreflightReport",
    "PreflightSeverity",
    "PreflightStatus",
    "analytical_preflight",
    "apply_adaptation_plan",
    "derive_adaptation_plan",
    "empirical_preflight",
    # version
    "__version__",
]
