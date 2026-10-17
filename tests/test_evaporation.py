import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aquacrop_jax.evaporation import (extract_water_from_evap_layer,
                                     evaporation_reduction_coefficient,
                                     stage1_evaporation, stage2_evaporation)
from aquacrop_jax.soil import profile_water_mm


def test_extract_water_visits_surface_first_and_conserves_mass():
    initial = jnp.asarray([0.29, 0.29], jnp.float64)
    result = extract_water_from_evap_layer(
        initial, [0.1, 0.1], [0.0, 0.0], 13.0, 5.0, 0.10)
    assert float(result.extracted) == pytest.approx(5.0)
    assert np.asarray(result.theta) == pytest.approx([0.24, 0.29])
    before = profile_water_mm(initial, [0.1, 0.1], [0.0, 0.0])
    after = profile_water_mm(result.theta, [0.1, 0.1], [0.0, 0.0])
    assert float(before - after) == pytest.approx(5.0)


def test_extraction_stops_at_air_dry_limit():
    result = extract_water_from_evap_layer(
        [0.065], [0.1], [0.0], 13.0, 5.0, 0.10)
    assert float(result.extracted) == pytest.approx(0.0)
    assert np.asarray(result.theta) == pytest.approx([0.065])


def test_stage1_is_limited_by_rew_store_and_clears_it():
    result = stage1_evaporation(
        [0.29], [0.1], [0.0], 13.0, 10.0, 7.0, 0.10)
    assert float(result.actual_evaporation) == pytest.approx(7.0)
    assert float(result.remaining_surface_water) == pytest.approx(0.0)
    assert float(result.remaining_potential) == pytest.approx(3.0)


def test_stage2_reduction_coefficient_has_fortran_limits():
    values = np.asarray(evaporation_reduction_coefficient([0.0, 0.5, 1.0], 4.0))
    assert values[0] == pytest.approx(0.0)
    assert values[2] == pytest.approx(1.0)
    assert 0.0 < values[1] < 1.0


def test_stage2_evaporation_conserves_water_and_depth_bounds():
    initial = jnp.asarray([0.25, 0.25, 0.25], jnp.float64)
    result = stage2_evaporation(
        initial, [0.1, 0.1, 0.1], [0.0, 0.0, 0.0],
        46.0, 29.0, 13.0, 7.0, 0.0, 5.0)
    before = profile_water_mm(initial, [0.1, 0.1, 0.1], [0.0, 0.0, 0.0])
    after = profile_water_mm(result.theta, [0.1, 0.1, 0.1], [0.0, 0.0, 0.0])
    assert float(before - after) == pytest.approx(float(result.actual_evaporation))
    assert 0.15 <= float(result.evaporation_depth) <= 0.30
    assert float(result.actual_evaporation) <= 5.0


def test_stage2_inner_depth_loop_traces_with_symbolic_scalar():
    @jax.jit
    def traced(potential):
        return stage2_evaporation(
            jnp.asarray([0.20, 0.20, 0.20], jnp.float64),
            jnp.asarray([0.10, 0.10, 0.10], jnp.float64),
            jnp.zeros((3,), jnp.float64), 46.0, 29.0, 13.0,
            7.0, 0.0, potential).evaporation_depth

    depth = traced(jnp.asarray(3.0, jnp.float64))
    assert depth.shape == ()
    assert 0.15 <= float(depth) <= 0.30
