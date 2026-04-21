"""Walk-forward generalization gate tests."""

from __future__ import annotations

from omegaprompt.core.walkforward import evaluate_walk_forward


def test_walk_forward_passes_on_small_gap():
    wf = evaluate_walk_forward(0.80, 0.75, max_gap=0.25)
    assert wf.passed is True
    assert abs(wf.generalization_gap - 0.0625) < 1e-6


def test_walk_forward_fails_on_large_gap():
    wf = evaluate_walk_forward(0.80, 0.40, max_gap=0.25)
    assert wf.passed is False


def test_walk_forward_fails_on_zero_train():
    wf = evaluate_walk_forward(0.0, 0.0)
    # 0/0 is reported as 1.0 gap; fails any finite max_gap.
    assert wf.passed is False


def test_walk_forward_kc4_pearson_computed_on_shared_ids():
    per_item_train = {"t1": 0.9, "t2": 0.8, "t3": 0.7, "t4": 0.6}
    per_item_test = {"t1": 0.85, "t2": 0.75, "t3": 0.7, "t4": 0.55}  # monotonic
    wf = evaluate_walk_forward(
        0.75, 0.71,
        per_item_train=per_item_train,
        per_item_test=per_item_test,
        max_gap=0.10,
        min_kc4=0.5,
    )
    assert wf.kc4_correlation is not None
    assert wf.kc4_correlation > 0.9
    assert wf.passed is True


def test_walk_forward_kc4_failure_fails_even_with_good_gap():
    # Train has variance but test rankings are anti-correlated.
    per_item_train = {"t1": 0.1, "t2": 0.3, "t3": 0.5, "t4": 0.7, "t5": 0.9}
    per_item_test = {"t1": 0.9, "t2": 0.7, "t3": 0.5, "t4": 0.3, "t5": 0.1}
    wf = evaluate_walk_forward(
        0.5, 0.5,
        per_item_train=per_item_train,
        per_item_test=per_item_test,
        max_gap=1.0,
        min_kc4=0.9,
    )
    assert wf.kc4_correlation is not None
    assert wf.kc4_correlation < 0
    assert wf.passed is False


def test_walk_forward_kc4_skipped_when_train_variance_zero():
    # Zero variance on train -> pearson undefined -> kc4 is None.
    per_item_train = {"t1": 1.0, "t2": 1.0, "t3": 1.0}
    per_item_test = {"t1": 0.5, "t2": 0.7, "t3": 0.3}
    wf = evaluate_walk_forward(
        0.9, 0.9,
        per_item_train=per_item_train,
        per_item_test=per_item_test,
        max_gap=1.0,
        min_kc4=0.9,
    )
    assert wf.kc4_correlation is None
    # None kc4 means the gate is inert; we only fail on the gap check.
    assert wf.passed is True


def test_walk_forward_skips_kc4_when_too_few_shared_ids():
    wf = evaluate_walk_forward(
        0.8, 0.75,
        per_item_train={"t1": 0.5},
        per_item_test={"t1": 0.55},
    )
    assert wf.kc4_correlation is None
    assert wf.passed is True
