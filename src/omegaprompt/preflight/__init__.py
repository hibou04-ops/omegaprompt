"""Preflight layer - analytical and empirical checks that run before the
main calibration pipeline to adapt run parameters to the actual environment.

Two complementary sub-units compose into a shared :class:`AdaptationPlan`:

- :mod:`omegaprompt.preflight.mini_antemortem` (analytical) reads the run
  configuration and emits REAL / GHOST / NEW / UNRESOLVED classifications
  over a small set of calibration-specific trap patterns (self-agreement
  bias, small-sample KC-4 power, subjective rubric dimensions, ...).
- :mod:`omegaprompt.preflight.mini_omega_lock` (empirical) issues small
  probe calls to measure judge consistency, endpoint schema reliability,
  context budget margin, latency, and noise floor.

Both feed into :func:`derive_adaptation_plan`, which returns an
:class:`AdaptationPlan` the main ``omegacal calibrate`` command consumes
to adjust its thresholds. The discipline (hard_gate x soft_score, walk-
forward, sensitivity unlock) is preserved - only the numeric parameters
(`min_kc4`, `max_gap`, `unlock_k`, `rescore_count`, rubric weights,
schema fallback) adapt.

The full runtime versions of both sub-units ship in their own repositories
(``mini-omega-lock``, ``mini-antemortem-cli``) for reuse across
calibration domains beyond prompts. This module provides the in-process
minimal implementations the core ``omegacal`` pipeline needs to stand
alone.
"""

from omegaprompt.preflight.adaptation import (
    AdaptationPlan,
    ParameterOverride,
    derive_adaptation_plan,
    apply_adaptation_plan,
)
from omegaprompt.preflight.contracts import (
    AnalyticalFinding,
    PreflightReport,
    PreflightSeverity,
    PreflightStatus,
)
from omegaprompt.preflight.mini_antemortem import (
    analytical_preflight,
    analytical_traps,
)
from omegaprompt.preflight.mini_omega_lock import empirical_preflight

__all__ = [
    "AdaptationPlan",
    "AnalyticalFinding",
    "ParameterOverride",
    "PreflightReport",
    "PreflightSeverity",
    "PreflightStatus",
    "analytical_preflight",
    "analytical_traps",
    "apply_adaptation_plan",
    "derive_adaptation_plan",
    "empirical_preflight",
]
