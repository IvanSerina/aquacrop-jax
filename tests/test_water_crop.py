import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax import lax

from aquacrop_jax.salinity import initialize_salt, salt_geometry
from aquacrop_jax.soil import profile_water_mm
from aquacrop_jax.water_crop import (run_water_crop_trajectory,
                                     water_crop_pre_et_step)


def _trajectory(theta, rain, root, tpot):
    return run_water_crop_trajectory(
        theta, rain, [4.0] * len(rain), root, tpot,
        [0.1, 0.1, 0.1], [0.0, 0.0, 0.0], [0.29, 0.29, 0.29],
        46.0, 29.0, 13.0, 1.0, 1200.0, 46, 7.0, 0.60,
        0.020, 0.010, stomatal_shape=3.0)


def test_water_crop_trajectory_closes_balance_with_both_losses():
    result = _trajectory([0.29, 0.29, 0.29], [2.0, 0.0, 5.0],
                         [0.2, 0.2, 0.25], [2.0, 3.0, 2.0])
    assert result.theta.shape == (3, 3)
    assert sum(np.asarray(result.evaporation)) > 0.0
    assert sum(np.asarray(result.transpiration)) > 0.0
    assert np.asarray(result.residual) == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_zero_root_depth_disables_transpiration():
    result = _trajectory([0.29, 0.29, 0.29], [0.0], [0.0], [5.0])
    assert np.asarray(result.transpiration) == pytest.approx([0.0])


def test_daily_feedback_computes_et_demand_after_pre_et_water_state():
    theta = jnp.asarray([.14, .16, .16], jnp.float64)
    eto = jnp.asarray([4., 4.], jnp.float64)

    def feedback(state, pre_et, forcing):
        eto_day, = forcing
        water_before_et = jnp.mean(pre_et.theta)
        tpot = eto_day * water_before_et
        epot = eto_day * (1.0 - water_before_et)
        return state + 1, tpot, epot, (water_before_et, tpot, epot)

    result = jax.jit(lambda: run_water_crop_trajectory(
        theta, [40., 0.], eto, [.2, .2], None,
        [.1, .1, .1], [0., 0., 0.], [.29, .29, .29],
        46., 29., 13., 1., 1200., 46, 7., .6, .02, .01,
        stomatal_shape=3., daily_feedback_step=feedback,
        initial_ece=1., irrigation_ec=[0., 0.], groundwater_ec=[0., 0.],
        initial_feedback_state=jnp.int32(0), feedback_forcing=(eto,)))()
    assert result.feedback[0].shape == (2,)
    assert np.asarray(result.feedback[0])[0] > np.asarray(result.feedback[0])[1]
    np.testing.assert_allclose(result.feedback[1], eto * result.feedback[0])
    np.testing.assert_allclose(result.feedback[2], eto * (1. - result.feedback[0]))
    np.testing.assert_allclose(result.water.residual, 0., rtol=0., atol=1e-9)
    np.testing.assert_allclose(result.water.salt_residual, 0., rtol=0., atol=1e-9)
    assert np.all(np.asarray(result.water.transpiration) > 0.)


def test_daily_irrigation_enters_profile_and_balance():
    result = run_water_crop_trajectory(
        [0.20, 0.20, 0.20], [0.0], [0.0], [0.0], [0.0],
        [0.1, 0.1, 0.1], [0.0, 0.0, 0.0], [0.29, 0.29, 0.29],
        46.0, 29.0, 13.0, 1.0, 1200.0, 46, 7.0, 0.60,
        0.020, 0.010, daily_irrigation=[12.0])
    assert np.asarray(result.irrigation) == pytest.approx([12.0])
    assert np.asarray(result.infiltration) == pytest.approx([12.0])
    assert np.asarray(result.residual) == pytest.approx([0.0], abs=1e-9)


def test_groundwater_capillary_rise_is_balanced():
    result = run_water_crop_trajectory(
        [0.20, 0.20, 0.20], [0.0], [0.0], [0.5], [0.0],
        [0.2, 0.2, 0.2], [0.0] * 3, [0.29] * 3,
        46.0, 29.0, 13.0, 1.0, 100.0, 46, 7.0, 0.60,
        0.020, 0.010, groundwater_depth_m=[1.0],
        capillary_a=[-0.3] * 3, capillary_b=[1.0] * 3,
        minimum_root_depth_m=0.1)
    assert float(result.capillary_rise[0]) >= 0.0
    assert float(result.groundwater_depth[0]) == 1.0
    assert np.asarray(result.residual) == pytest.approx([0.0], abs=1e-8)


def test_saline_recurrence_conserves_salt_and_restarts_from_explicit_state():
    from aquacrop_jax.salinity import SaltState
    common = dict(thickness=[.1, .1], gravel_volume_pct=[0., 0.],
        adjusted_fc=[.29, .29], saturation_pct=46., field_capacity_pct=29.,
        wilting_point_pct=13., tau=1., ksat=1200., curve_number=46,
        rew_mm=7., p_stomatal=.6, smax_top=.02, smax_bottom=.01)
    forcing = dict(initial_theta=[.2, .2], rain=[0., 0.], eto=[0., 0.],
        rooting_depth=[.2, .2], potential_transpiration=[0., 0.],
        daily_irrigation=[10., 0.], irrigation_ec=[5., 0.])
    result = run_water_crop_trajectory(**forcing, **common)
    assert np.asarray(result.salt_infiltrated) == pytest.approx([32., 0.])
    assert np.asarray(result.salt_residual) == pytest.approx([0., 0.], abs=1e-11)
    assert np.asarray(result.residual) == pytest.approx([0., 0.], abs=1e-10)
    first = run_water_crop_trajectory(**{k: v[:1] if k != 'initial_theta' else v
        for k, v in forcing.items()}, **common)
    second = run_water_crop_trajectory(first.theta[-1], [0.], [0.], [.2], [0.],
        initial_salt=SaltState(first.salt_dissolved[-1], first.salt_deposited[-1]),
        initial_surface_water_mm=first.evap_surface_water[-1],
        initial_evap_depth_m=first.evap_depth[-1],
        initial_stage2_start_pct=first.stage2_start[-1], **common)
    assert np.asarray(second.salt_dissolved[-1]) == pytest.approx(
        np.asarray(result.salt_dissolved[-1]), abs=1e-11)


def test_salt_stress_limits_transpiration_without_removing_salt():
    common = dict(initial_theta=[.25, .25], rain=[0.], eto=[0.],
        rooting_depth=[.2], potential_transpiration=[2.],
        thickness=[.1, .1], gravel_volume_pct=[0., 0.], adjusted_fc=[.29, .29],
        saturation_pct=46., field_capacity_pct=29., wilting_point_pct=13., tau=1.,
        ksat=1200., curve_number=46, rew_mm=7., p_stomatal=.6,
        smax_top=.02, smax_bottom=.01, salt_stomatal_coefficients=(0., 1., 0.))
    fresh = run_water_crop_trajectory(initial_ece=0., **common)
    salty = run_water_crop_trajectory(initial_ece=14., **common)
    assert float(salty.transpiration[0]) < float(fresh.transpiration[0])
    assert float(salty.salt_residual[0]) == pytest.approx(0., abs=1e-10)
    assert np.asarray(salty.ece[0]) == pytest.approx([14., 14.])


def test_saline_groundwater_inflow_enters_salt_and_water_budgets():
    result = run_water_crop_trajectory(
        [.20, .20], [0.], [0.], [.2], [0.], [.1, .1], [0., 0.], [.29, .29],
        46., 29., 13., 1., 100., 46, 7., .6, .02, .01,
        groundwater_depth_m=[.10], groundwater_ec=[5.],
        capillary_a=[-.3, -.3], capillary_b=[1., 1.])
    water_in = float(result.groundwater_inflow[0] + result.capillary_rise[0])
    assert water_in > 0.
    assert float(result.salt_groundwater[0]) == pytest.approx(water_in * 5. * .64)
    assert float(result.salt_residual[0]) == pytest.approx(0., abs=1e-10)
    assert float(result.residual[0]) == pytest.approx(0., abs=1e-10)


def test_salinity_stress_average_survives_recurrent_restart():
    from aquacrop_jax.salinity import SaltState
    common = dict(thickness=[.1, .1], gravel_volume_pct=[0., 0.],
        adjusted_fc=[.29, .29], saturation_pct=46., field_capacity_pct=29.,
        wilting_point_pct=13., tau=1., ksat=1200., curve_number=46, rew_mm=7.,
        p_stomatal=.6, smax_top=.02, smax_bottom=.01,
        salt_stomatal_coefficients=(0., 1., 0.))
    full = run_water_crop_trajectory([.25, .25], [0., 0., 0.], [0.] * 3,
        [.2] * 3, [2.] * 3, initial_ece=8., salinity_growing_mask=[True, False, True], **common)
    first = run_water_crop_trajectory([.25, .25], [0.], [0.], [.2], [2.],
        initial_ece=8., salinity_growing_mask=[True], **common)
    last = run_water_crop_trajectory(first.theta[-1], [0., 0.], [0., 0.], [.2, .2], [2., 2.],
        initial_salt=SaltState(first.salt_dissolved[-1], first.salt_deposited[-1]),
        initial_surface_water_mm=first.evap_surface_water[-1],
        initial_evap_depth_m=first.evap_depth[-1],
        initial_stage2_start_pct=first.stage2_start[-1],
        initial_salinity_growing_days=first.salinity_growing_days[-1],
        initial_mean_salinity_stress_pct=first.mean_salinity_stress_pct[-1],
        salinity_growing_mask=[False, True], **common)
    assert np.asarray(full.salinity_growing_days) == pytest.approx([1., 1., 2.])
    assert np.asarray(full.salinity_stomatal_reduction_pct) == pytest.approx([43., 0., 43.])
    np.testing.assert_allclose(last.theta, full.theta[1:], atol=1e-12)
    np.testing.assert_allclose(last.mean_salinity_stress_pct, full.mean_salinity_stress_pct[1:], atol=1e-12)


def test_pre_et_boundary_conserves_water_and_salt_with_twelve_compartments():
    thickness = jnp.full((12,), 0.1, jnp.float64)
    gravel = jnp.zeros((12,), jnp.float64)
    saturation = 46.0
    field_capacity = 29.0
    ksat = 1200.0
    geometry = salt_geometry(jnp.full((12,), saturation, jnp.float64),
                             jnp.full((12,), field_capacity, jnp.float64),
                             jnp.full((12,), ksat, jnp.float64), 20.0)
    theta = jnp.full((12,), 0.25, jnp.float64)
    salt = initialize_salt(1.0, theta, thickness, gravel, saturation, geometry, 100.0)

    def step(carry, forcing):
        current_theta, current_salt = carry
        rain, irrigation, irrigation_ec = forcing
        pre = water_crop_pre_et_step(
            current_theta, current_salt, rain, irrigation, -1.0, 0.0,
            irrigation_ec, thickness=thickness, gravel_volume_pct=gravel,
            saturation_pct=saturation, field_capacity_pct=field_capacity,
            wilting_point_pct=13.0,
            adjusted_fc=jnp.full((12,), field_capacity / 100.0, jnp.float64),
            tau=1.0, ksat=ksat, curve_number=46,
            initial_abstraction_pct=5.0, capillary_enabled=False,
            cap_a=jnp.zeros_like(theta), cap_b=jnp.zeros_like(theta),
            salt_enabled=True, geometry=geometry, salt_solubility=100.0)
        return (pre.theta, pre.salt), pre

    forcing = (jnp.asarray([0.0, 2.0]), jnp.asarray([10.0, 0.0]),
               jnp.asarray([5.0, 0.0]))
    _, outputs = jax.jit(lambda: lax.scan(step, (theta, salt), forcing))()
    assert outputs.theta.shape == (2, 12)
    assert outputs.salt.dissolved.shape == (2, 12, 11)
    water_before = jnp.concatenate((profile_water_mm(theta, thickness, gravel)[None],
                                    jax.vmap(profile_water_mm, in_axes=(0, None, None))(
                                        outputs.theta[:-1], thickness, gravel)))
    water_after = jax.vmap(profile_water_mm, in_axes=(0, None, None))(
        outputs.theta, thickness, gravel)
    water_residual = water_before + forcing[0] + forcing[1] - outputs.water.runoff - \
        outputs.water.drainage - water_after
    salt_before = jnp.concatenate((jnp.sum(salt.dissolved + salt.deposited)[None],
        jnp.sum(outputs.salt.dissolved[:-1] + outputs.salt.deposited[:-1], axis=(1, 2))))
    salt_after = jnp.sum(outputs.salt.dissolved + outputs.salt.deposited, axis=(1, 2))
    salt_residual = salt_before + outputs.salt_infiltrated_g_m2 - \
        outputs.salt_drained_g_m2 - salt_after
    np.testing.assert_allclose(water_residual, 0.0, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(salt_residual, 0.0, rtol=0.0, atol=1e-9)
