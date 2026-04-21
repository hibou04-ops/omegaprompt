"""Walk-forward train/test generalization assessment.

The contract:

1. Search picks ``best_params`` on the training slice.
2. We replay ``best_params`` on the held-out test slice.
3. We compute ``generalization_gap = |train - test| / |train|`` and, if
   the slices share item ids, a Pearson correlation between per-item
   scores (KC-4 gate). The threshold is declared up front and not
   adjustable post-hoc.

If generalization_gap exceeds the caller's ``max_gap`` OR the KC-4
correlation falls below ``min_kc4``, the candidate fails and the artifact
status flips to ``FAIL_KC4_GATE``.
"""

from __future__ import annotations

from math import isnan

from omegaprompt.domain.result import WalkForwardResult


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = (var_x * var_y) ** 0.5
    if denom == 0:
        return None
    r = cov / denom
    if isnan(r):
        return None
    return r


def evaluate_walk_forward(
    train_best_fitness: float,
    test_fitness: float,
    *,
    per_item_train: dict[str, float] | None = None,
    per_item_test: dict[str, float] | None = None,
    max_gap: float = 0.25,
    min_kc4: float = 0.5,
) -> WalkForwardResult:
    """Compute walk-forward result + pass/fail verdict.

    Parameters
    ----------
    train_best_fitness, test_fitness:
        Aggregate fitness on each slice for the chosen ``best_params``.
    per_item_train, per_item_test:
        Optional per-item score maps. If both are provided and share at
        least 3 ids, KC-4 Pearson correlation is computed over the shared
        ids. Otherwise kc4 is left None.
    max_gap:
        Maximum allowed ``|train - test| / |train|``. Above this the
        candidate fails.
    min_kc4:
        Minimum Pearson correlation required when kc4 is computable.
        Below this the candidate fails. If kc4 is None (slices do not
        share ids), this check is skipped.
    """
    if train_best_fitness == 0:
        gap = float("inf")
    else:
        gap = abs(train_best_fitness - test_fitness) / abs(train_best_fitness)

    kc4: float | None = None
    if per_item_train and per_item_test:
        shared_ids = sorted(set(per_item_train) & set(per_item_test))
        if len(shared_ids) >= 3:
            xs = [per_item_train[k] for k in shared_ids]
            ys = [per_item_test[k] for k in shared_ids]
            kc4 = _pearson(xs, ys)

    gap_ok = gap <= max_gap
    kc4_ok = kc4 is None or kc4 >= min_kc4
    passed = gap_ok and kc4_ok

    return WalkForwardResult(
        train_best_fitness=train_best_fitness,
        test_fitness=test_fitness,
        generalization_gap=gap if gap != float("inf") else 1.0,
        kc4_correlation=kc4,
        passed=passed,
    )
