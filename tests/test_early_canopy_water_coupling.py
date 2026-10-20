"""Day-order proof for the early canopy, ET and conservative water/salt steps."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aquacrop_jax.canopy import (EarlyVegetativeCanopyState,
                                WaterSenescenceState,
                                calculate_potential_et,
                                early_vegetative_gdd_step,
                                early_water_senescence_gdd_step,
                                leaf_thresholds_for_eto)
from aquacrop_jax.roots import canopy_water_zone
from aquacrop_jax.water_crop import run_water_crop_trajectory


@pytest.mark.parametrize("include_senescence", [False, True])
def test_rain_relaxes_leaf_stress_before_et_and_both_balances_close(include_senescence):
    thickness = jnp.full((3,), .1, jnp.float64)
    gravel = jnp.zeros((3,), jnp.float64)
    # Lower compartments start slightly wetter, so the root zone wins the
    # native strict comparison before rain wets the top compartment.
    theta = jnp.asarray([.14, .16, .16], jnp.float64)
    canopy = EarlyVegetativeCanopyState(
        jnp.float64(.08), jnp.float64(.05), jnp.float64(.08), jnp.bool_(False))
    senescence = WaterSenescenceState(
        jnp.float64(0.), jnp.float64(0.), jnp.float64(.08), jnp.float64(0.),
        jnp.bool_(False), jnp.bool_(False))

    def feedback(state, pre_et, forcing):
        canopy_state, senescence_state = state
        total_gdd, daily_gdd, dap, eto = forcing
        zone = canopy_water_zone(pre_et.theta, thickness, gravel,
                                 .3, 29., 13., 46., .1)
        p_upper, p_lower = leaf_thresholds_for_eto(eto, .15, .55)
        next_canopy, stress, valid = early_vegetative_gdd_step(
            canopy_state, total_gdd, daily_gdd, 5., 600., 1920., 1920., 1920.,
            .05, .95, .012, .006, 24., 40., zone.actual_mm,
            zone.field_capacity_mm, zone.wilting_point_mm,
            p_upper, p_lower, 3.)
        if include_senescence:
            next_canopy, senescence_state, _, valid = early_water_senescence_gdd_step(
                senescence_state, next_canopy, canopy_state.cover,
                total_gdd, daily_gdd, 5., 1920., 1920., .05, .05, .95, .95 * .6,
                .006, zone.actual_mm, zone.field_capacity_mm, zone.wilting_point_mm,
                eto, .65, 3., 40., canopy_valid=valid)
        demand = calculate_potential_et(
            dap, 1, 40, 180, 180, next_canopy.cover, eto, 1.15, 11.,
            .95, senescence_state.withered_cover if include_senescence else next_canopy.cover,
            60., 400., 10., 8.,
            evaporation_limit=senescence_state.evaporation_limited if include_senescence else True)
        tpot = jnp.where(valid, demand.transpiration_mm, jnp.nan)
        epot = jnp.where(valid, demand.evaporation_mm, jnp.nan)
        diagnostics = (next_canopy.cover, stress, valid, zone.topsoil_considered,
                      pre_et.theta, senescence_state.time_gdd,
                      senescence_state.withered_cover)
        return (next_canopy, senescence_state), tpot, epot, diagnostics

    gdd = jnp.asarray([70., 80.])
    daily_gdd = jnp.asarray([10., 10.])
    dap = jnp.asarray([6., 7.])
    eto = jnp.asarray([4., 4.])
    result = jax.jit(lambda: run_water_crop_trajectory(
        theta, [0., 40.], eto, [.3, .3], None,
        thickness, gravel, [.29, .29, .29], 46., 29., 13., 1., 1200., 46, 7.,
        .6, .02, .01, stomatal_shape=3., initial_ece=1.,
        maximum_evap_depth_m=.3, decline_factor=4.,
        daily_feedback_step=feedback, initial_feedback_state=(canopy, senescence),
        feedback_forcing=(gdd, daily_gdd, dap, eto)))()
    (cover, stress, valid, topsoil, pre_theta,
     senescence_time, withered) = result.feedback
    np.testing.assert_array_equal(valid, [True, True])
    np.testing.assert_array_equal(topsoil, [False, True])
    assert float(stress[0]) == 100.
    if include_senescence:
        assert float(cover[0]) < .08
        np.testing.assert_array_equal(senescence_time, [10., 0.])
        assert float(withered[0]) == .08
        assert float(withered[1]) == float(cover[1])
    else:
        assert float(cover[0]) == .08
    assert float(stress[1]) < 100.
    assert float(cover[1]) > float(cover[0])
    assert float(pre_theta[1, 0]) > float(pre_theta[0, 0])
    np.testing.assert_allclose(result.water.residual, 0., rtol=0., atol=1e-9)
    np.testing.assert_allclose(result.water.salt_residual, 0., rtol=0., atol=1e-9)
