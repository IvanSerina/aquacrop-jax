"""Rain reaches the full minimum root zone before germination is checked."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from aquacrop_jax.crop_clock import (CropClockState, accumulate_crop_gdd,
    check_germination_step, seeded_canopy_clock)
from aquacrop_jax.roots import root_zone_water, canopy_water_zone
from aquacrop_jax.salinity import SaltState
from aquacrop_jax.water_crop import run_water_crop_trajectory

THICKNESS = jnp.full(3, .1, jnp.float64)
DAYS = jnp.arange(100, 103, dtype=jnp.int32)
RAIN = jnp.asarray([0., 10., 0.], jnp.float64)


class Diagnostics(NamedTuple):
    clock: object
    state: CropClockState
    root_water: jax.Array
    topsoil_considered_for_canopy: jax.Array


def initial():
    return CropClockState(jnp.bool_(False), jnp.bool_(False), jnp.int32(0),
                          jnp.float64(0.), jnp.float64(0.))


def simulate(theta, days, rain, state, *, salt=None, surface=0., depth=.15, stage=-1.):
    def pre(state, water, day):
        # AdvanceOneTimeStep captures this local clock before CheckGermination.
        state = accumulate_crop_gdd(state, 5., day, 100, 102)
        clock = seeded_canopy_clock(state, day, 100)
        root = root_zone_water(water.theta, THICKNESS, jnp.zeros(3), .3, 30., 10., 46.)
        canopy_zone = canopy_water_zone(water.theta, THICKNESS, jnp.zeros(3),
                                        .3, 30., 10., 46., .1)
        state = check_germination_step(state, day, 100, root.actual_mm,
            root.field_capacity_mm, root.wilting_point_mm, 25)
        # ET held at zero to isolate the water-input/germination boundary.
        return state, jnp.float64(0.), jnp.float64(0.), Diagnostics(
            clock, state, root.actual_mm, canopy_zone.topsoil_considered)

    return run_water_crop_trajectory(theta, rain, jnp.zeros_like(rain),
        jnp.full_like(rain, .3), None, THICKNESS, jnp.zeros(3), jnp.full(3, .30),
        46., 30., 10., 1., 1200., 46, 7., .6, .02, .01, stomatal_shape=3.,
        initial_ece=1. if salt is None else None, initial_salt=salt,
        initial_surface_water_mm=surface, initial_evap_depth_m=depth,
        initial_stage2_start_pct=stage, daily_feedback_step=pre,
        initial_feedback_state=state, feedback_forcing=days)


def test_germination_uses_full_root_zone_and_rain_of_the_same_day():
    result = jax.jit(simulate)(jnp.asarray([.2, .1, .1]), DAYS, RAIN, initial())
    diagnostic = result.feedback
    assert bool(diagnostic.topsoil_considered_for_canopy[0])
    assert float(diagnostic.root_water[0]) < 45.
    np.testing.assert_array_equal(diagnostic.state.germinated, [False, True, True])
    np.testing.assert_array_equal(diagnostic.state.delayed_days, [1, 1, 1])
    np.testing.assert_array_equal(diagnostic.clock.virtual_day, [0, 0, 1])
    np.testing.assert_array_equal(diagnostic.clock.cumulative_gdd, [5., 5., 10.])
    np.testing.assert_array_equal(diagnostic.state.sum_gdd, [0., 5., 10.])
    np.testing.assert_array_equal(diagnostic.state.sum_gdd_from_day1, [5., 10., 15.])
    assert float(diagnostic.root_water[1]) >= 45.
    np.testing.assert_allclose(result.water.residual, 0., rtol=0., atol=1e-9)
    np.testing.assert_allclose(result.water.salt_residual, 0., rtol=0., atol=1e-9)


def test_germination_state_and_water_salt_restart_on_the_rain_day():
    theta = jnp.asarray([.2, .1, .1])
    operation = jax.jit(simulate)
    whole = operation(theta, DAYS, RAIN, initial())
    first = operation(theta, DAYS[:1], RAIN[:1], initial())
    water = first.water
    # Pre-ET-only trajectories expose the carry in this explicit diagnostic.
    # No crop state changes after the callback in this isolated germination test.
    first_clock = jax.tree.map(lambda values: values[-1], first.feedback.state)
    second = operation(water.theta[-1], DAYS[1:], RAIN[1:], first_clock,
        salt=SaltState(water.salt_dissolved[-1], water.salt_deposited[-1]),
        surface=water.evap_surface_water[-1], depth=water.evap_depth[-1], stage=water.stage2_start[-1])
    for a, b in zip(second.feedback.state, whole.feedback.state):
        np.testing.assert_array_equal(a[-1], b[-1])
    np.testing.assert_allclose(second.water.theta[-1], whole.water.theta[-1], rtol=0., atol=1e-14)
    np.testing.assert_allclose(second.water.salt_dissolved[-1], whole.water.salt_dissolved[-1], rtol=0., atol=1e-14)
    np.testing.assert_array_equal(second.feedback.state.germinated, whole.feedback.state.germinated[1:])
