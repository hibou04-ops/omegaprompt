# SPDX-License-Identifier: Apache-2.0
"""Offline contracts for the additive calibration verdict helper."""

import math

import pytest

from omegaprompt.runtime import verify_calibration_preflight


@pytest.mark.parametrize(
    ("observed_delta", "noise_floor_mde", "expected"),
    [
        (0.2, 0.1, "ESTABLISHED"),
        (0.1, 0.1, "NOT_ESTABLISHED"),
        (0.05, 0.1, "NOT_ESTABLISHED"),
        (0.0, 0.0, "NOT_ESTABLISHED"),
        (-0.1, 0.0, "NOT_ESTABLISHED"),
    ],
)
def test_verdict_requires_delta_strictly_above_noise_floor(
    observed_delta, noise_floor_mde, expected
):
    result = verify_calibration_preflight(observed_delta, noise_floor_mde)
    assert result == {
        "verdict": expected,
        "observed_delta": observed_delta,
        "noise_floor_mde": noise_floor_mde,
        "blocking_traps": [],
    }


@pytest.mark.parametrize("observed_delta", [0.01, 0.1, 1.0])
def test_blocking_traps_prevent_established_verdict(observed_delta):
    result = verify_calibration_preflight(observed_delta, 0.0, ("OMG001",))
    assert result["verdict"] == "NOT_ESTABLISHED"
    assert result["blocking_traps"] == ["OMG001"]


def test_default_noise_floor_preserves_positive_delta_use():
    assert verify_calibration_preflight(0.1)["verdict"] == "ESTABLISHED"


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


@pytest.mark.parametrize("observed_delta", [math.nan, math.inf, -math.inf])
def test_nonfinite_delta_is_rejected(observed_delta):
    with pytest.raises(ValueError, match="observed_delta"):
        verify_calibration_preflight(observed_delta)


@pytest.mark.parametrize("noise_floor_mde", [-0.1, math.nan, math.inf, -math.inf])
def test_invalid_noise_floor_is_rejected(noise_floor_mde):
    with pytest.raises(ValueError, match="noise_floor_mde"):
        verify_calibration_preflight(0.2, noise_floor_mde)
