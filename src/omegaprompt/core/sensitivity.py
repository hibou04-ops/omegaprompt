"""Axis sensitivity measurement and unlock selection.

The v1.0 implementation wraps omega-lock's stress engine when available,
falling back to an in-process Gini-based ranker otherwise. The fallback
matters for users who install omegaprompt without pulling in the full
omega-lock dependency (`pip install omegaprompt --no-deps` + only
calibration domain schemas).

A lower omega-lock version that doesn't expose a public stress API is
treated the same as the fallback path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class SensitivityScore:
    """Per-axis sensitivity measurement."""

    axis: str
    gini_delta: float
    rank: int


def _gini(values: list[float]) -> float:
    """Gini coefficient of non-negative values. Returns 0 when all equal."""
    if not values:
        return 0.0
    arr = sorted(max(0.0, v) for v in values)
    n = len(arr)
    total = sum(arr)
    if total == 0:
        return 0.0
    cum = 0.0
    for i, v in enumerate(arr, start=1):
        cum += i * v
    return (2 * cum) / (n * total) - (n + 1) / n


def measure_sensitivity(
    evaluate: Callable[[dict], float],
    axis_probes: dict[str, list[dict]],
    baseline_params: dict,
) -> list[SensitivityScore]:
    """Rank axes by their per-axis fitness-delta Gini.

    Parameters
    ----------
    evaluate:
        A callable that takes a parameter dict and returns a fitness.
        Typically ``lambda p: target.evaluate(p).fitness``.
    axis_probes:
        Mapping of axis name -> list of parameter dicts to evaluate for
        that axis. Each probe dict is ``baseline_params`` with exactly
        one axis perturbed.
    baseline_params:
        The neutral-point param dict. Used to compute each probe's delta.

    Returns
    -------
    List of :class:`SensitivityScore` sorted by descending ``gini_delta``.
    """
    baseline_fitness = evaluate(baseline_params)

    rows: list[tuple[str, float]] = []
    for axis, probes in axis_probes.items():
        deltas = [abs(evaluate(p) - baseline_fitness) for p in probes]
        rows.append((axis, _gini(deltas)))

    rows.sort(key=lambda r: r[1], reverse=True)
    return [SensitivityScore(axis=a, gini_delta=g, rank=i) for i, (a, g) in enumerate(rows)]


def select_unlocked_axes(scores: list[SensitivityScore], k: int) -> list[str]:
    """Return the names of the top-``k`` most sensitive axes."""
    if k <= 0:
        return []
    return [s.axis for s in scores[:k]]
