import pytest
import jax.numpy as jnp

from aquacrop_jax.hydrology import (daily_water_step, run_water_trajectory,
                                   vertical_infiltration)
from aquacrop_jax.soil import profile_water_mm


def test_infiltration_fills_top_down_and_conserves_mass():
    theta0 = jnp.asarray([0.20, 0.20], jnp.float64)
    result = vertical_infiltration(theta0, [0.1, 0.1], [0.0, 0.0],
                                   46.0, 30.0, 1200.0)
    assert result.theta == pytest.approx([0.46, 0.24])
    assert result.deep_percolation == pytest.approx(0.0)
    assert result.rejected == pytest.approx(0.0)
    before = profile_water_mm(theta0, [0.1, 0.1], [0.0, 0.0])
    after = profile_water_mm(result.theta, [0.1, 0.1], [0.0, 0.0])
    assert after - before == pytest.approx(30.0)


def test_infiltration_reports_deep_percolation_after_saturation():
    result = vertical_infiltration([0.46, 0.46], [0.1, 0.1], [0.0, 0.0],
                                   46.0, 8.0, 1200.0)
    assert result.theta == pytest.approx([0.46, 0.46])
    assert result.deep_percolation == pytest.approx(8.0)


def test_daily_water_step_closes_mass_balance():
    result = daily_water_step(
        [0.29, 0.29], [0.1, 0.1], [0.0, 0.0], [0.29, 0.29],
        46.0, 29.0, 1.0, 1200.0, 20.0, 46, 5.0)
    assert result.residual.dtype == jnp.float64
    assert result.residual == pytest.approx(0.0, abs=1e-10)
    assert result.infiltration + result.runoff == pytest.approx(20.0)


def test_water_trajectory_propagates_state_and_closes_every_day():
    trajectory = run_water_trajectory(
        [0.29, 0.29], [0.0, 20.0, 0.0], [0.1, 0.1], [0.0, 0.0],
        [0.29, 0.29], 46.0, 29.0, 1.0, 1200.0, 46, 5.0)
    assert trajectory.theta.shape == (3, 2)
    assert trajectory.theta[0] == pytest.approx([0.29, 0.29])
    assert trajectory.theta[1] != pytest.approx(trajectory.theta[0])
    assert trajectory.theta[2] != pytest.approx(trajectory.theta[1])
    assert trajectory.residual == pytest.approx([0.0, 0.0, 0.0], abs=1e-10)


def test_water_trajectory_flux_vectors_have_one_value_per_day():
    rain = jnp.asarray([0.0, 1.0, 2.0, 3.0], jnp.float64)
    trajectory = run_water_trajectory(
        [0.29], rain, [0.1], [0.0], [0.29],
        46.0, 29.0, 1.0, 1200.0, 46)
    assert trajectory.runoff.shape == rain.shape
    assert trajectory.drainage.shape == rain.shape
    assert trajectory.infiltration.shape == rain.shape
    assert trajectory.residual.shape == rain.shape


def test_water_trajectory_preserves_static_compartment_shape():
    trajectory = run_water_trajectory(
        jnp.full((12,), .29, jnp.float64), [0.1, 0.2],
        jnp.full((12,), .1, jnp.float64),
        jnp.zeros((12,), jnp.float64),
        jnp.full((12,), .29, jnp.float64),
        46.0, 29.0, 1.0, 1200.0, 46)
    assert trajectory.theta.shape == (2, 12)
