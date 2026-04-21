"""Preflight plugin interface.

This package is the thin *contract layer* between the main ``omegaprompt``
calibration pipeline and two optional external sub-tools:

- **mini-omega-lock** (separate PyPI package / repository) - empirical
  preflight: probe calls that measure judge consistency, endpoint schema
  reliability, context-budget margin, latency, noise floor.
- **mini-antemortem-cli** (separate PyPI package / repository) -
  analytical preflight: deterministic classifier over calibration trap
  patterns (self-agreement bias, small-sample KC-4 power, rubric weight
  concentration, etc.).

Both sub-tools emit a :class:`PreflightReport` (see
:mod:`omegaprompt.preflight.contracts`). The main pipeline consumes that
report via :func:`derive_adaptation_plan`, which produces an
:class:`AdaptationPlan` whose overrides only *strengthen* the default
discipline parameters (see :func:`apply_adaptation_plan` for the clipping
invariants).

**Standalone users do not need either sub-tool.** ``omegaprompt`` runs
end-to-end with the pipeline's declared defaults. The preflight
interface lights up only when a caller explicitly constructs a
``PreflightReport`` - typically by importing one of the external
sub-tools alongside ``omegaprompt``::

    pip install omegaprompt                 # main calibration only
    pip install omegaprompt mini-omega-lock mini-antemortem-cli   # full preflight
"""

from omegaprompt.preflight.adaptation import (
    AdaptationPlan,
    ParameterOverride,
    apply_adaptation_plan,
    derive_adaptation_plan,
)
from omegaprompt.preflight.contracts import (
    AnalyticalFinding,
    EndpointMeasurement,
    JudgeQualityMeasurement,
    PerformanceMeasurement,
    PreflightReport,
    PreflightSeverity,
    PreflightStatus,
)

__all__ = [
    # interface types
    "AdaptationPlan",
    "AnalyticalFinding",
    "EndpointMeasurement",
    "JudgeQualityMeasurement",
    "ParameterOverride",
    "PerformanceMeasurement",
    "PreflightReport",
    "PreflightSeverity",
    "PreflightStatus",
    # adaptation functions
    "apply_adaptation_plan",
    "derive_adaptation_plan",
]
