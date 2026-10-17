import jax
import numpy as np
import pytest

from aquacrop_jax.hydrology import daily_water_step, daily_water_step_irrigated
from aquacrop_jax.irrigation import (generated_irrigation, manual_irrigation,
                                    net_irrigation)


def test_manual_schedule_sums_matching_events_and_jits():
    scheduled = jax.jit(manual_irrigation)
    assert float(scheduled(12, [10, 12, 12], [5.0, 8.0, 2.0])) == 10.0
    assert float(scheduled(11, [10, 12], [5.0, 8.0])) == 0.0


def test_generated_raw_fixed_and_refill_modes():
    fixed = generated_irrigation(70.0, 100.0, 60.0, 50.0, 25.0)
    assert bool(fixed.triggered)
    assert float(fixed.trigger_depletion_mm) == pytest.approx(20.0)
    assert float(fixed.depth_mm) == pytest.approx(25.0)
    refill = generated_irrigation(90.0, 100.0, 60.0, 50.0, 5.0,
                                  depth_mode="refill")
    assert not bool(refill.triggered)
    assert float(refill.depth_mm) == 0.0


def test_net_irrigation_changes_only_rooted_fraction_and_conserves_depth():
    result = net_irrigation(
        [0.15, 0.15, 0.15], [0.1, 0.1, 0.1], [0.0, 0.0, 0.0],
        [29.0, 29.0, 29.0], [0.13, 0.13, 0.13], 0.25, 50.0)
    assert np.asarray(result.theta)[2] < np.asarray(result.theta)[1]
    before = 100.0 * 0.45
    after = 100.0 * float(np.sum(np.asarray(result.theta)))
    assert after - before == pytest.approx(float(result.depth_mm))


def test_zero_irrigation_matches_existing_water_step():
    args = ([0.29, 0.29], [0.1, 0.1], [0.0, 0.0], [0.29, 0.29],
            46.0, 29.0, 1.0, 1200.0, 20.0)
    old = daily_water_step(*args, 46, 5.0)
    new = daily_water_step_irrigated(*args, 0.0, 46, 5.0)
    for left, right in zip(old, new):
        assert np.asarray(left) == pytest.approx(np.asarray(right), abs=1e-12)


def test_irrigated_water_step_closes_balance():
    result = daily_water_step_irrigated(
        [0.20, 0.20], [0.1, 0.1], [0.0, 0.0], [0.29, 0.29],
        46.0, 29.0, 1.0, 1200.0, 0.0, 15.0, 46)
    assert float(result.residual) == pytest.approx(0.0, abs=1e-10)
    assert float(result.infiltration + result.runoff) == pytest.approx(15.0)
