import jax.numpy as jnp
import numpy as np
import pytest

from aquacrop_jax.water_evap import (prepare_stage2_percentage,
                                    run_water_evap_trajectory)


def test_prepare_stage2_percentage_at_field_capacity():
    percentage = prepare_stage2_percentage(
        [0.29, 0.29], [0.1, 0.1], [0.0, 0.0], 46.0, 29.0, 7.0)
    assert percentage.dtype == jnp.float64
    assert 0.0 <= float(percentage) <= 100.0


def test_coupled_water_evaporation_trajectory_closes_daily_balance():
    trajectory = run_water_evap_trajectory(
        [0.29, 0.29, 0.29], [2.0, 0.0, 5.0], [3.0, 4.0, 2.0],
        [0.1, 0.1, 0.1], [0.0, 0.0, 0.0], [0.29, 0.29, 0.29],
        46.0, 29.0, 13.0, 1.0, 1200.0, 46, 7.0)
    assert trajectory.theta.shape == (3, 3)
    assert trajectory.evaporation.shape == (3,)
    assert (np.asarray(trajectory.evaporation) >= 0.0).all()
    assert np.asarray(trajectory.residual) == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
