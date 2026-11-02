# SPDX-License-Identifier: Apache-2.0
"""Offline contracts for the additive calibration verdict helper."""

import math

import pytest

from omegaprompt.runtime import verify_calibration_preflight


@pytest.mark.parametrize(
    ("observed_delta", "noise_floor_mde", "expected"),
    [
        (0.2, 0.1, "PREFLIGHT_VALIDATED"),
        (0.1, 0.1, "NOT_ESTABLISHED"),
        (0.05, 0.1, "NOT_ESTABLISHED"),
        (0.0, 0.0, "NOT_ESTABLISHED"),
        (0.2, 0.0, "NOT_ESTABLISHED"),
        (0.2, -0.0, "NOT_ESTABLISHED"),
        (-0.1, 0.0, "NOT_ESTABLISHED"),
    ],
)
def test_verdict_requires_delta_strictly_above_noise_floor(
    observed_delta, noise_floor_mde, expected
):
    result = verify_calibration_preflight(observed_delta, noise_floor_mde)
    assert result == {
        "verdict": expected,
        "preflight_status": (
            "PREFLIGHT_VALIDATED" if expected == "PREFLIGHT_VALIDATED" else "PREFLIGHT_REJECTED"
        ),
        "noise_floor_measured": True,
        "holdout_verified": False,
        "deployment_authorized": False,
        "observed_delta": observed_delta,
        "noise_floor_mde": noise_floor_mde,
        "blocking_traps": [],
    }


@pytest.mark.parametrize("observed_delta", [0.01, 0.1, 1.0])
def test_blocking_traps_prevent_preflight_validation(observed_delta):
    result = verify_calibration_preflight(observed_delta, 0.001, ("OMG001",))
    assert result["verdict"] == "NOT_ESTABLISHED"
    assert result["preflight_status"] == "PREFLIGHT_REJECTED"
    assert result["blocking_traps"] == ["OMG001"]


@pytest.mark.parametrize("blocking_traps", [(), ("OMG001",)])
def test_omitted_and_none_noise_floor_are_unmeasured_and_block_validation(blocking_traps):
    omitted = verify_calibration_preflight(1.0, blocking_traps=blocking_traps)
    explicit_none = verify_calibration_preflight(1.0, None, blocking_traps)
    assert omitted == explicit_none
    assert omitted["noise_floor_mde"] is None
    assert omitted["noise_floor_measured"] is False
    assert omitted["verdict"] == "NOT_ESTABLISHED"
    assert omitted["preflight_status"] == "PREFLIGHT_UNMEASURED_NOISE"
    assert omitted["holdout_verified"] is False
    assert omitted["deployment_authorized"] is False


def test_explicit_zero_is_supplied_but_does_not_validate_preflight():
    result = verify_calibration_preflight(1.0, 0.0)
    assert result["noise_floor_measured"] is True
    assert result["preflight_status"] == "PREFLIGHT_REJECTED"
    assert result["verdict"] == "NOT_ESTABLISHED"


@pytest.mark.parametrize("noise_floor_mde", [1e-300, 0.1, 1.0])
def test_supplied_positive_noise_requires_a_strictly_larger_delta(noise_floor_mde):
    result = verify_calibration_preflight(noise_floor_mde * 2.0, noise_floor_mde)
    assert result["noise_floor_measured"] is True
    assert result["preflight_status"] == result["verdict"] == "PREFLIGHT_VALIDATED"
    assert verify_calibration_preflight(noise_floor_mde, noise_floor_mde)["verdict"] == "NOT_ESTABLISHED"


@pytest.mark.parametrize("observed_delta", [-1.0, 0.0, 0.1, 1e308])
@pytest.mark.parametrize("noise_floor_mde", [None, 0.0, 0.1, 1e308])
@pytest.mark.parametrize("blocking_traps", [(), ("OMG001",)])
def test_preflight_never_verifies_holdout_or_authorizes_deployment(
    observed_delta, noise_floor_mde, blocking_traps
):
    result = verify_calibration_preflight(observed_delta, noise_floor_mde, blocking_traps)
    assert result["holdout_verified"] is False
    assert result["deployment_authorized"] is False
    assert result["verdict"] != "ESTABLISHED"
    assert (result["preflight_status"] == "PREFLIGHT_VALIDATED") == (
        noise_floor_mde is not None and noise_floor_mde > 0.0
        and not blocking_traps and observed_delta > noise_floor_mde
    )


def test_result_does_not_alias_or_mutate_callers_traps():
    traps = ["OMG002", "OMG001"]
    result = verify_calibration_preflight(1.0, blocking_traps=traps)
    assert traps == ["OMG002", "OMG001"]
    assert result["blocking_traps"] == traps
    result["blocking_traps"].append("OMG003")
    assert traps == ["OMG002", "OMG001"]


def test_repeated_calls_return_equal_independent_results():
    first = verify_calibration_preflight(0.2, 0.1)
    second = verify_calibration_preflight(0.2, 0.1)
    assert first == second
    assert first is not second
    assert first["blocking_traps"] is not second["blocking_traps"]


@pytest.mark.parametrize("observed_delta", [math.nan, math.inf, -math.inf, True, False, "0.2"])
def test_nonfinite_delta_is_rejected(observed_delta):
    with pytest.raises(ValueError, match="observed_delta"):
        verify_calibration_preflight(observed_delta)


@pytest.mark.parametrize("noise_floor_mde", [-0.1, math.nan, math.inf, -math.inf, True, False, "0.1"])
def test_invalid_noise_floor_is_rejected(noise_floor_mde):
    with pytest.raises(ValueError, match="noise_floor_mde"):
        verify_calibration_preflight(0.2, noise_floor_mde)
