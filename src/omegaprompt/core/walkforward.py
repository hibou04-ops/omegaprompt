"""Walk-forward train/test generalization assessment.

The contract:

1. Search picks ``best_params`` on the training slice.
2. We replay ``best_params`` on the held-out test slice.
3. We compute ``generalization_gap = |train - test| / |train|``. The
   threshold is declared up front and not adjustable post-hoc.
4. KC-4 (Pearson correlation between per-item train/test scores) is
   only computed when the two slices share item ids. This is by design:
   a "paired" replay reuses the same items on both sides so per-item
   correlation is well-defined. An ordinary disjoint train/test split
   has no paired items, so KC-4 is structurally unmeasurable on that
   shape and the gate falls back to the gap-only check.

KC-4 semantics by validation_mode
---------------------------------

- ``"auto"`` (default, backward-compat): if the slices share >=3 item
  ids, compute KC-4 and apply ``min_kc4``. Otherwise leave ``kc4``
  None and skip the correlation check. This matches the historical
  behaviour and is the right default for callers who pass per-item
  maps without thinking about pairing.
- ``"paired"``: the caller asserts that the two slices share item
  ids by design. If fewer than 3 ids overlap, raise rather than
  silently skipping — a paired run that produces no KC-4 is a setup
  bug, not a pass.
- ``"disjoint"``: the caller asserts that the two slices have
  disjoint item ids by design (the standard held-out split). KC-4
  is not computed; the gate is gap-only. The artifact records this
  fact so downstream readers know KC-4 was not skipped, it was
  structurally unmeasurable on the chosen split.

If generalization_gap exceeds ``max_gap`` OR (when KC-4 is computed)
the correlation falls below ``min_kc4``, the candidate fails and the
artifact status flips to ``FAIL_KC4_GATE``.
"""

from __future__ import annotations

from math import isnan
from typing import Literal

from omegaprompt.domain.result import WalkForwardResult

ValidationMode = Literal["auto", "paired", "disjoint"]


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
    validation_mode: ValidationMode = "auto",
) -> WalkForwardResult:
    """Compute walk-forward result + pass/fail verdict.

    Parameters
    ----------
    train_best_fitness, test_fitness:
        Aggregate fitness on each slice for the chosen ``best_params``.
    per_item_train, per_item_test:
        Optional per-item score maps. KC-4 correlation is computed
        from shared ids; see ``validation_mode``.
    max_gap:
        Maximum allowed ``|train - test| / |train|``. Above this the
        candidate fails.
    min_kc4:
        Minimum Pearson correlation required when KC-4 is computable.
    validation_mode:
        One of ``"auto" | "paired" | "disjoint"``. See module docstring.
        Default ``"auto"`` preserves historical behaviour: KC-4 is
        computed only when slices share >=3 ids, otherwise skipped.
    """
    if train_best_fitness == 0:
        gap = float("inf")
    else:
        gap = abs(train_best_fitness - test_fitness) / abs(train_best_fitness)

    shared_ids: list[str] = []
    if per_item_train and per_item_test:
        shared_ids = sorted(set(per_item_train) & set(per_item_test))

    kc4: float | None = None
    if validation_mode == "disjoint":
        # Caller asserts disjoint split; do not compute KC-4 even if
        # ids happen to overlap (e.g. id collision across splits is a
        # bug to surface elsewhere, not a free pass for the gate).
        pass
    elif validation_mode == "paired":
        if len(shared_ids) < 3:
            raise ValueError(
                "validation_mode='paired' requires per_item_train and "
                "per_item_test to share at least 3 item ids, got "
                f"{len(shared_ids)}. A paired run with no overlap is a "
                "setup bug; either pair the slices on the same items or "
                "use validation_mode='disjoint'."
            )
        xs = [per_item_train[k] for k in shared_ids]
        ys = [per_item_test[k] for k in shared_ids]
        kc4 = _pearson(xs, ys)
    else:  # "auto"
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
