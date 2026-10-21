"""Seeded emergence/protection drives actual ET after daily water inputs."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aquacrop_jax.canopy import calculate_potential_et
from aquacrop_jax.roots import root_zone_water, canopy_water_zone
from aquacrop_jax.rounding import round_fortran
from aquacrop_jax.salinity import SaltState
from aquacrop_jax.seeded_crop import (begin_seeded_crop_day,
    germinate_seeded_crop_step, seeded_canopy_step)
from aquacrop_jax.thermal_time import calendar_days_from_record
from aquacrop_jax.water_crop import run_water_crop_trajectory
from test_seeded_crop import initial, PARAMETERS, EFFECT

THICKNESS = jnp.full(3, .1, jnp.float64)
DAYS = jnp.arange(100, 104, dtype=jnp.int32)
HEAT = jnp.asarray([5., 5., 30., 5.])
RAIN = jnp.asarray([0., 20., 0., 0.])
RECORD = jnp.concatenate((HEAT, jnp.full(200, 5., jnp.float64)))


class Diagnostic(NamedTuple):
    state: object
    clock: object
    valid: jax.Array
    root_water: jax.Array
    demand: object


def simulate(theta, days, heat, rain, state, *, salt=None, surface=0., depth=.15, stage=-1.):
    def before(state, water, forcing):
        day, daily = forcing
        state, clock = begin_seeded_crop_day(state, day, daily, 100, 99, 103, 1, .05)
        root = root_zone_water(water.theta, THICKNESS, jnp.zeros(3), .3, 30., 10., 46.)
        state = germinate_seeded_crop_step(state, day, 100, root.actual_mm,
            root.field_capacity_mm, root.wilting_point_mm, 25)
        zone = canopy_water_zone(water.theta, THICKNESS, jnp.zeros(3), .3, 30., 10., 46., .1)
        state, diagnostic = seeded_canopy_step(state, clock, daily, 2., zone.actual_mm,
            zone.field_capacity_mm, zone.wilting_point_mm, PARAMETERS, EFFECT)
        # Native ET uses base full-canopy timing and its own thermal DAP,
        # adjusted maximum cover, current withered cover and evaporation flag.
        converted = calendar_days_from_record(RECORD, round_fortran(clock.cumulative_gdd))
        demand = calculate_potential_et(converted.value + state.delayed_days,
            *PARAMETERS.calendar_days, state.canopy.canopy.cover, 2., 1.15, 11.,
            state.canopy.canopy.maximum_adjusted, state.canopy.senescence.withered_cover,
            60., 369.41, daily, 8., delayed_days=state.delayed_days, senescence_exponent=1.,
            evaporation_limit=state.canopy.senescence.evaporation_limited)
        return state, demand.transpiration_mm, demand.evaporation_mm, Diagnostic(
            state, clock, diagnostic.valid & converted.valid, root.actual_mm, demand)

    return run_water_crop_trajectory(theta, rain, jnp.full_like(rain, 2.),
        jnp.full_like(rain, .3), None, THICKNESS, jnp.zeros(3), jnp.full(3, .30),
        46., 30., 10., 1., 1200., 46, 7., .6, .02, .01, stomatal_shape=3.,
        initial_ece=1. if salt is None else None, initial_salt=salt,
        initial_surface_water_mm=surface, initial_evap_depth_m=depth,
        initial_stage2_start_pct=stage, daily_feedback_step=before,
        initial_feedback_state=state, feedback_forcing=(days, heat))


@pytest.fixture(scope='module')
def whole():
    return jax.jit(simulate)(jnp.asarray([.2, .1, .1]), DAYS, HEAT, RAIN, initial())


def test_seeded_emergence_and_protection_drive_transpiration_after_rain(whole):
    r = whole
    np.testing.assert_array_equal(r.feedback.valid, True)
    np.testing.assert_array_equal(r.feedback.state.germinated, [False, True, True, True])
    np.testing.assert_array_equal(r.feedback.state.clock.protected_seedling, [False, True, False, False])
    np.testing.assert_array_equal(r.feedback.state.delayed_days, [1, 1, 1, 1])
    np.testing.assert_array_equal(r.feedback.clock.cumulative_gdd, [5., 5., 35., 40.])
    np.testing.assert_array_equal(r.feedback.state.sum_gdd_from_day1, [5., 10., 40., 45.])
    np.testing.assert_array_equal(r.feedback.demand.transpiration_mm[:2], 0.)
    assert np.all(np.asarray(r.feedback.demand.transpiration_mm[2:]) > 0.)
    assert np.all(np.asarray(r.water.transpiration[2:]) > 0.)
    assert np.all(np.asarray(r.water.evaporation) > 0.)
    np.testing.assert_allclose(r.water.residual, 0., rtol=0., atol=1e-9)
    np.testing.assert_allclose(r.water.salt_residual, 0., rtol=0., atol=1e-9)


def test_restart_while_seedling_is_protected_preserves_actual_et_and_balances(whole):
    operation = jax.jit(simulate)
    first = operation(jnp.asarray([.2, .1, .1]), DAYS[:2], HEAT[:2], RAIN[:2], initial())
    state = jax.tree.map(lambda x: x[-1], first.feedback.state)
    assert bool(state.clock.protected_seedling)
    water = first.water
    second = operation(water.theta[-1], DAYS[2:], HEAT[2:], RAIN[2:], state,
        salt=SaltState(water.salt_dissolved[-1], water.salt_deposited[-1]),
        surface=water.evap_surface_water[-1], depth=water.evap_depth[-1], stage=water.stage2_start[-1])
    for a, b in zip(jax.tree.leaves(second.feedback.state), jax.tree.leaves(whole.feedback.state)):
        np.testing.assert_allclose(a, b[2:], rtol=0., atol=1e-14)
    np.testing.assert_allclose(second.water.theta[-1], whole.water.theta[-1], rtol=0., atol=1e-14)
    np.testing.assert_allclose(second.water.salt_dissolved[-1], whole.water.salt_dissolved[-1], rtol=0., atol=1e-14)
    np.testing.assert_allclose(second.water.transpiration, whole.water.transpiration[2:], rtol=0., atol=1e-14)
    np.testing.assert_allclose(second.water.residual, 0., rtol=0., atol=1e-9)
    np.testing.assert_allclose(second.water.salt_residual, 0., rtol=0., atol=1e-9)
