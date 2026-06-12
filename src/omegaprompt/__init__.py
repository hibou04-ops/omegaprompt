"""omegaprompt - provider-neutral prompt calibration engine.

v1.1 is the public-distribution release of the v1.0 architecture. The
*discipline* (sensitivity-driven coordinate descent, hard_gate x
soft_score fitness, walk-forward ship gate, machine-readable artifacts)
is preserved intact; the *contract* is provider-neutral: meta-axes
instead of Claude-specific knobs, a single ``call()`` method on
:class:`LLMProvider`, a :class:`Judge` protocol with rule / LLM /
ensemble implementations.

The main package is self-contained. Two optional sub-tools
(``mini-omega-lock``, ``mini-antemortem-cli``) distribute separately
and plug in via the ``omegaprompt.preflight`` interface to add empirical
and analytical preflight measurements respectively. Standalone users
do not need them.
"""

from omegaprompt.core import (
    CompositeFitness,
    GateResult,
    OverfitMetrics,
    assess_run_risk,
    aggregate_fitness,
    evaluate_walk_forward,
    extract_overfit_metrics,
    item_fitness,
    load_artifact,
    measure_sensitivity,
    overfit_metrics_dict,
    policy_for,
    relaxed_safeguards_for,
    render_gate_report,
    run_gate,
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
    EndpointMeasurement,
    JudgeQualityMeasurement,
    ParameterOverride,
    PerformanceMeasurement,
    PreflightReport,
    PreflightSeverity,
    PreflightStatus,
    apply_adaptation_plan,
    derive_adaptation_plan,
)
from omegaprompt.targets import CalibrableTarget, PromptTarget
from omegaprompt.runtime import (
    ArtifactDiff,
    CalibrateTuning,
    GradeResult,
    ProviderSpec,
    SensitivityResult,
    SensitivityTuning,
    calibrate,
    classify_traps,
    diff,
    evaluate,
    grade,
    measure_sensitivity,
    preflight,
    report,
)

__version__ = "2.1.0"

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
    # overfit surfacing (2.1.0) — train<->holdout transfer metrics
    "OverfitMetrics",
    "extract_overfit_metrics",
    "overfit_metrics_dict",
    # ship gate (2.1.0) — integrity + holdout transfer/gap verdict
    "GateResult",
    "run_gate",
    "render_gate_report",
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
    # runtime — high-level Tier 1 + Tier 2 entrypoints (agent-callable surface)
    "ArtifactDiff",
    "CalibrateTuning",
    "GradeResult",
    "ProviderSpec",
    "SensitivityResult",
    "SensitivityTuning",
    "calibrate",
    "classify_traps",
    "diff",
    "evaluate",
    "grade",
    "measure_sensitivity",
    "preflight",
    "report",
    # preflight (plugin interface for optional mini-* sub-tools)
    "AdaptationPlan",
    "AnalyticalFinding",
    "EndpointMeasurement",
    "JudgeQualityMeasurement",
    "ParameterOverride",
    "PerformanceMeasurement",
    "PreflightReport",
    "PreflightSeverity",
    "PreflightStatus",
    "apply_adaptation_plan",
    "derive_adaptation_plan",
    # version
    "__version__",
]
