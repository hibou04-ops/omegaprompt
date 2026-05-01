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


# ---------------------------------------------------------------------------
# validation_mode — paired vs disjoint vs auto.
# Reviewer P3: README claims KC-4 is the holdout gate. In reality KC-4 is
# only meaningful on a paired replay; on a disjoint train/test split it
# was silently skipped. The new validation_mode parameter forces callers
# to be explicit so neither side gets a hidden free pass.
# ---------------------------------------------------------------------------


import pytest


def test_validation_mode_disjoint_skips_kc4_even_with_overlap():
    """disjoint mode never computes KC-4, even if ids accidentally overlap."""
    per_item_train = {"t1": 0.9, "t2": 0.8, "t3": 0.7, "t4": 0.6}
    per_item_test = {"t1": 0.85, "t2": 0.75, "t3": 0.7, "t4": 0.55}
    wf = evaluate_walk_forward(
        0.75, 0.71,
        per_item_train=per_item_train,
        per_item_test=per_item_test,
        max_gap=0.10,
        min_kc4=0.5,
        validation_mode="disjoint",
    )
    assert wf.kc4_correlation is None
    assert wf.passed is True  # gap-only gate


def test_validation_mode_paired_raises_when_overlap_too_small():
    """A paired run with no overlap is a setup bug, not a free pass."""
    per_item_train = {"t1": 0.5, "t2": 0.6, "t3": 0.7}
    per_item_test = {"t10": 0.5, "t20": 0.6, "t30": 0.7}  # disjoint ids
    with pytest.raises(ValueError, match="paired.*at least 3"):
        evaluate_walk_forward(
            0.75, 0.70,
            per_item_train=per_item_train,
            per_item_test=per_item_test,
            validation_mode="paired",
        )


def test_validation_mode_paired_computes_kc4_normally():
    per_item_train = {"t1": 0.9, "t2": 0.8, "t3": 0.7, "t4": 0.6}
    per_item_test = {"t1": 0.85, "t2": 0.75, "t3": 0.7, "t4": 0.55}
    wf = evaluate_walk_forward(
        0.75, 0.71,
        per_item_train=per_item_train,
        per_item_test=per_item_test,
        max_gap=0.10,
        min_kc4=0.5,
        validation_mode="paired",
    )
    assert wf.kc4_correlation is not None
    assert wf.kc4_correlation > 0.9
    assert wf.passed is True


def test_validation_mode_auto_matches_legacy_behaviour():
    """auto is the default and matches pre-validation_mode behaviour."""
    per_item_train = {"t1": 0.9, "t2": 0.8, "t3": 0.7, "t4": 0.6}
    per_item_test = {"t1": 0.85, "t2": 0.75, "t3": 0.7, "t4": 0.55}
    wf_default = evaluate_walk_forward(
        0.75, 0.71,
        per_item_train=per_item_train,
        per_item_test=per_item_test,
        max_gap=0.10,
        min_kc4=0.5,
    )
    wf_auto = evaluate_walk_forward(
        0.75, 0.71,
        per_item_train=per_item_train,
        per_item_test=per_item_test,
        max_gap=0.10,
        min_kc4=0.5,
        validation_mode="auto",
    )
    assert wf_default.kc4_correlation == wf_auto.kc4_correlation
    assert wf_default.passed == wf_auto.passed


# ---------------------------------------------------------------------------
# Reviewer P1 #6: WalkForwardResult records *why* KC-4 / the gap have the
# values they do. Each kc4_status / gap_status branch needs a test that
# asserts the field is the documented value, not just that the legacy
# numeric output still works.
# ---------------------------------------------------------------------------


def test_kc4_status_computed_when_pearson_succeeds():
    per_item_train = {"t1": 0.9, "t2": 0.8, "t3": 0.7, "t4": 0.6}
    per_item_test = {"t1": 0.85, "t2": 0.75, "t3": 0.7, "t4": 0.55}
    wf = evaluate_walk_forward(
        0.75, 0.71,
        per_item_train=per_item_train,
        per_item_test=per_item_test,
        max_gap=0.10,
        min_kc4=0.5,
    )
    assert wf.kc4_status == "COMPUTED"
    assert wf.shared_item_count == 4
    assert wf.validation_mode == "auto"


def test_kc4_status_disjoint_records_not_applicable():
    per_item_train = {"t1": 0.9, "t2": 0.8, "t3": 0.7}
    per_item_test = {"t10": 0.5, "t20": 0.6, "t30": 0.7}
    wf = evaluate_walk_forward(
        0.75, 0.71,
        per_item_train=per_item_train,
        per_item_test=per_item_test,
        validation_mode="disjoint",
    )
    assert wf.kc4_status == "NOT_APPLICABLE_DISJOINT"
    assert wf.kc4_correlation is None
    assert wf.shared_item_count == 0
    assert wf.min_kc4_threshold is None  # threshold inert under disjoint


def test_kc4_status_insufficient_shared_ids_in_auto():
    wf = evaluate_walk_forward(
        0.8, 0.75,
        per_item_train={"t1": 0.5, "t2": 0.6},
        per_item_test={"t1": 0.55, "t2": 0.65},
    )
    assert wf.kc4_status == "INSUFFICIENT_SHARED_ITEMS"
    assert wf.kc4_correlation is None
    assert wf.shared_item_count == 2


def test_kc4_status_missing_per_item_scores():
    wf = evaluate_walk_forward(0.8, 0.75)
    assert wf.kc4_status == "MISSING_PER_ITEM_SCORES"
    assert wf.kc4_correlation is None
    assert wf.shared_item_count == 0


def test_kc4_status_zero_variance_train_in_auto_still_passes():
    """Auto mode preserves backward-compat: zero-variance train -> KC-4
    None -> gate inert. The new schema records *why* via kc4_status so
    a reviewer is not lied to about the gate firing."""
    per_item_train = {"t1": 1.0, "t2": 1.0, "t3": 1.0}
    per_item_test = {"t1": 0.5, "t2": 0.7, "t3": 0.3}
    wf = evaluate_walk_forward(
        0.9, 0.9,
        per_item_train=per_item_train,
        per_item_test=per_item_test,
        max_gap=1.0,
        min_kc4=0.9,
    )
    assert wf.kc4_status == "ZERO_VARIANCE_TRAIN"
    assert wf.passed is True  # auto-mode legacy behaviour


def test_kc4_status_zero_variance_test():
    per_item_train = {"t1": 0.1, "t2": 0.5, "t3": 0.9}
    per_item_test = {"t1": 0.7, "t2": 0.7, "t3": 0.7}
    wf = evaluate_walk_forward(
        0.5, 0.7,
        per_item_train=per_item_train,
        per_item_test=per_item_test,
        max_gap=1.0,
        min_kc4=0.5,
    )
    assert wf.kc4_status == "ZERO_VARIANCE_TEST"


def test_kc4_status_zero_variance_both():
    per_item_train = {"t1": 0.5, "t2": 0.5, "t3": 0.5}
    per_item_test = {"t1": 0.7, "t2": 0.7, "t3": 0.7}
    wf = evaluate_walk_forward(
        0.5, 0.7,
        per_item_train=per_item_train,
        per_item_test=per_item_test,
        max_gap=1.0,
        min_kc4=0.5,
    )
    assert wf.kc4_status == "ZERO_VARIANCE_BOTH"


# ---------------------------------------------------------------------------
# Reviewer P1 #7: paired mode is strict. zero-variance / unmeasurable KC-4
# is a setup bug, not a free pass.
# ---------------------------------------------------------------------------


def test_validation_mode_paired_zero_variance_fails_closed():
    """The caller asserted KC-4 is meaningful by choosing paired. An
    unmeasurable correlation must fail the gate."""
    per_item_train = {"t1": 1.0, "t2": 1.0, "t3": 1.0}
    per_item_test = {"t1": 0.9, "t2": 0.8, "t3": 0.7}
    wf = evaluate_walk_forward(
        0.9, 0.9,
        per_item_train=per_item_train,
        per_item_test=per_item_test,
        max_gap=1.0,
        validation_mode="paired",
    )
    assert wf.passed is False
    assert wf.kc4_status == "ZERO_VARIANCE_TRAIN"
    assert wf.kc4_correlation is None


def test_validation_mode_paired_records_min_kc4_even_when_unmeasurable():
    """min_kc4 is still meaningful in paired mode (the threshold the
    user declared); the artifact records it so the reviewer can see
    what gate would have applied if KC-4 had been computable."""
    per_item_train = {"t1": 0.5, "t2": 0.5, "t3": 0.5}
    per_item_test = {"t1": 0.5, "t2": 0.5, "t3": 0.5}
    wf = evaluate_walk_forward(
        0.5, 0.5,
        per_item_train=per_item_train,
        per_item_test=per_item_test,
        min_kc4=0.7,
        validation_mode="paired",
    )
    assert wf.passed is False
    assert wf.min_kc4_threshold == 0.7


# ---------------------------------------------------------------------------
# Reviewer P1 #8: gap_status preserves the difference between
# "structurally undefined" (train=0, test=0) and "denominator zero,
# infinite uplift" (train=0, test!=0).
# ---------------------------------------------------------------------------


def test_gap_status_ok_for_normal_run():
    wf = evaluate_walk_forward(0.80, 0.75, max_gap=0.25)
    assert wf.gap_status == "OK"
    assert wf.passed is True


def test_gap_status_train_zero_both_zero():
    wf = evaluate_walk_forward(0.0, 0.0)
    assert wf.gap_status == "TRAIN_ZERO_BOTH_ZERO"
    assert wf.passed is False
    # gap is reported as 1.0 for JSON friendliness; the reason lives on status.
    assert wf.generalization_gap == 1.0


def test_gap_status_train_zero_test_nonzero():
    wf = evaluate_walk_forward(0.0, 0.5)
    assert wf.gap_status == "TRAIN_ZERO_TEST_NONZERO"
    assert wf.passed is False
    assert wf.generalization_gap == 1.0


# ---------------------------------------------------------------------------
# Threshold + shared_item_count round-trip through the artifact schema.
# Reviewer P1 #6: a future reader should be able to tell which
# max_gap / min_kc4 produced the verdict without rerunning.
# ---------------------------------------------------------------------------


def test_max_gap_and_min_kc4_thresholds_recorded():
    per_item_train = {"t1": 0.9, "t2": 0.8, "t3": 0.7, "t4": 0.6}
    per_item_test = {"t1": 0.85, "t2": 0.75, "t3": 0.7, "t4": 0.55}
    wf = evaluate_walk_forward(
        0.75, 0.71,
        per_item_train=per_item_train,
        per_item_test=per_item_test,
        max_gap=0.123,
        min_kc4=0.456,
    )
    assert wf.max_gap_threshold == 0.123
    assert wf.min_kc4_threshold == 0.456
    assert wf.shared_item_count == 4


def test_legacy_artifact_with_correlation_upgrades_to_computed():
    """An artifact written before kc4_status existed deserializes with
    the UNKNOWN_LEGACY default. The post-init validator must upgrade it
    to COMPUTED when kc4_correlation is non-None so the markdown report
    doesn't surface a contradictory state."""
    from omegaprompt.domain.result import WalkForwardResult

    legacy = WalkForwardResult(
        train_best_fitness=0.8,
        test_fitness=0.75,
        generalization_gap=0.0625,
        kc4_correlation=0.83,
        passed=True,
        # validation_mode, kc4_status, etc. all default
    )
    assert legacy.kc4_status == "COMPUTED"


def test_legacy_artifact_without_correlation_stays_unknown():
    """When the legacy artifact has no correlation either, leave the
    status as UNKNOWN_LEGACY — we genuinely don't know why."""
    from omegaprompt.domain.result import WalkForwardResult

    legacy = WalkForwardResult(
        train_best_fitness=0.8,
        test_fitness=0.75,
        generalization_gap=0.0625,
        passed=True,
    )
    assert legacy.kc4_status == "UNKNOWN_LEGACY"
    assert legacy.kc4_correlation is None


def test_computed_status_without_correlation_is_rejected():
    """The validator catches the inverse mistake at construction time:
    a status of COMPUTED with no correlation is contradictory."""
    import pytest
    from omegaprompt.domain.result import WalkForwardResult

    with pytest.raises(ValueError, match="kc4_status='COMPUTED' requires"):
        WalkForwardResult(
            train_best_fitness=0.8,
            test_fitness=0.75,
            generalization_gap=0.0625,
            kc4_correlation=None,
            kc4_status="COMPUTED",
            passed=True,
        )


def test_walkforward_result_round_trips_through_json():
    """All new fields must serialize and deserialize so an artifact
    written today still parses tomorrow."""
    from omegaprompt.domain.result import WalkForwardResult

    wf = evaluate_walk_forward(
        0.75, 0.71,
        per_item_train={"t1": 0.9, "t2": 0.8, "t3": 0.7, "t4": 0.6},
        per_item_test={"t1": 0.85, "t2": 0.75, "t3": 0.7, "t4": 0.55},
        max_gap=0.10,
        min_kc4=0.5,
        validation_mode="paired",
    )
    rt = WalkForwardResult.model_validate_json(wf.model_dump_json())
    assert rt.kc4_status == wf.kc4_status
    assert rt.gap_status == wf.gap_status
    assert rt.validation_mode == wf.validation_mode
    assert rt.shared_item_count == wf.shared_item_count
    assert rt.max_gap_threshold == wf.max_gap_threshold
    assert rt.min_kc4_threshold == wf.min_kc4_threshold
