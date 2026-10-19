"""Day-order proof for the early canopy, ET and conservative water/salt steps."""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from aquacrop_jax.canopy import (EarlyVegetativeCanopyState,
                                calculate_potential_et,
                                early_vegetative_gdd_step,
                                leaf_thresholds_for_eto)
from aquacrop_jax.roots import canopy_water_zone
from aquacrop_jax.salinity import initialize_salt, salt_geometry
from aquacrop_jax.water_crop import water_crop_pre_et_step, water_crop_post_et_step
from aquacrop_jax.water_evap import prepare_stage2_percentage


def test_rain_relaxes_leaf_stress_before_et_and_both_balances_close():
    thickness = jnp.full((3,), .1, jnp.float64)
    gravel = jnp.zeros((3,), jnp.float64)
    # Lower compartments start slightly wetter, so the root zone wins the
    # native strict comparison before rain wets the top compartment.
    theta = jnp.asarray([.14, .16, .16], jnp.float64)
    geometry = salt_geometry(jnp.full((3,), 46., jnp.float64),
                             jnp.full((3,), 29., jnp.float64),
                             jnp.full((3,), 1200., jnp.float64), 20.)
    salt = initialize_salt(1., theta, thickness, gravel, 46., geometry, 100.)
    stage2 = prepare_stage2_percentage(theta, thickness, gravel,
                                       46., 29., 7., .15)
    canopy = EarlyVegetativeCanopyState(
        jnp.float64(.08), jnp.float64(.05), jnp.float64(.08), jnp.bool_(False))

    def day(carry, forcing):
        (water, store, depth, stage, salt_state, salt_days, salt_mean,
         canopy_state) = carry
        total_gdd, rain, dap = forcing
        pre = water_crop_pre_et_step(
            water, salt_state, rain, 0., -1., 0., 0.,
            thickness=thickness, gravel_volume_pct=gravel,
            saturation_pct=46., field_capacity_pct=29., wilting_point_pct=13.,
            adjusted_fc=jnp.full((3,), .29, jnp.float64), tau=1.,
            ksat=1200., curve_number=46, initial_abstraction_pct=5.,
            capillary_enabled=False, cap_a=jnp.zeros((3,), jnp.float64),
            cap_b=jnp.zeros((3,), jnp.float64), salt_enabled=True,
            geometry=geometry, salt_solubility=100.)
        zone = canopy_water_zone(pre.theta, thickness, gravel,
                                 .3, 29., 13., 46., .1)
        p_upper, p_lower = leaf_thresholds_for_eto(4., .15, .55)
        next_canopy, stress, valid = early_vegetative_gdd_step(
            canopy_state, total_gdd, 10., 5., 600., 1920., 1920., 1920.,
            .05, .95, .012, .006, 24., 40., zone.actual_mm,
            zone.field_capacity_mm, zone.wilting_point_mm,
            p_upper, p_lower, 3.)
        demand = calculate_potential_et(
            dap, 1, 40, 180, 180, next_canopy.cover, 4., 1.15, 11.,
            .95, next_canopy.cover, 60., 400., 10., 8.)
        post, result = water_crop_post_et_step(
            pre, store, depth, stage, salt_days, salt_mean, rain, .3,
            demand.transpiration_mm, 0., -1., demand.evaporation_mm, True,
            thickness=thickness, gravel_volume_pct=gravel,
            saturation_pct=46., field_capacity_pct=29., wilting_point_pct=13.,
            rew_mm=7., maximum_evap_depth_m=.3, decline_factor=4.,
            minimum_root_depth_m=.1, anaerobic_sensitive=True,
            p_stomatal=.6, smax_top=.02, smax_bottom=.01,
            stomatal_shape=3., stress_reduction_pct=0., sink_correction=1.,
            salt_stomatal_coefficients=None, salinity_minimum_ece=2.,
            salinity_maximum_ece=16., salinity_response_pct=100.,
            salt_solubility=100., salt_enabled=True, geometry=geometry)
        return (*post, next_canopy), (next_canopy.cover, stress, valid,
                                      zone.topsoil_considered,
                                      pre.theta, result.residual,
                                      result.salt_residual)

    initial = (theta, jnp.float64(0.), jnp.float64(.15), stage2,
               salt, jnp.float64(0.), jnp.float64(0.), canopy)
    forcing = (jnp.asarray([70., 80.]), jnp.asarray([0., 40.]),
               jnp.asarray([6., 7.]))
    _, (cover, stress, valid, topsoil, pre_theta,
        water_residual, salt_residual) = (
        jax.jit(lambda: lax.scan(day, initial, forcing))())
    np.testing.assert_array_equal(valid, [True, True])
    np.testing.assert_array_equal(topsoil, [False, True])
    assert float(stress[0]) == 100.
    assert float(cover[0]) == .08
    assert float(stress[1]) < 100.
    assert float(cover[1]) > float(cover[0])
    assert float(pre_theta[1, 0]) > float(pre_theta[0, 0])
    np.testing.assert_allclose(water_residual, 0., rtol=0., atol=1e-9)
    np.testing.assert_allclose(salt_residual, 0., rtol=0., atol=1e-9)
