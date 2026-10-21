"""Seeded crop clock and post-infiltration germination state transitions."""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class CropClockState(NamedTuple):
    germinated: jax.Array
    protected_seedling: jax.Array
    delayed_days: jax.Array
    sum_gdd: jax.Array
    sum_gdd_from_day1: jax.Array


class SeededCanopyClock(NamedTuple):
    virtual_day: jax.Array
    cumulative_gdd: jax.Array


def accumulate_crop_gdd(state, daily_gdd, day_number, crop_first_day, simulation_last_day):
    """SetGDDVariablesNextDay accumulation, given the already computed heat.

    Both sums include today's heat before AdvanceOneTimeStep. Germination
    may reset sum_gdd later; sum_gdd_from_day1 always retains that heat.
    Initializing a simulation mid-crop is a separate native code path.
    """
    day, first, last = [jnp.asarray(v, jnp.int32) for v in
                        (day_number, crop_first_day, simulation_last_day)]
    heat = jnp.where((day >= first) & (day <= last), jnp.asarray(daily_gdd, jnp.float64), 0.)
    return state._replace(sum_gdd=jnp.asarray(state.sum_gdd, jnp.float64) + heat,
        sum_gdd_from_day1=jnp.asarray(state.sum_gdd_from_day1, jnp.float64) + heat)


def seeded_canopy_clock(state, day_number, crop_first_day):
    """AdvanceOneTimeStep seeded/transplanted GDD clock, before BUDGET.

    Capture this value before checking germination. Native BUDGET consumes
    these local values even when CheckGermination subsequently changes the
    persistent delay and sum_gdd. Regrowth has a separate adjusted clock.
    """
    virtual = (jnp.asarray(day_number, jnp.int32) -
        jnp.asarray(state.delayed_days, jnp.int32) - jnp.asarray(crop_first_day, jnp.int32))
    return SeededCanopyClock(virtual, jnp.asarray(state.sum_gdd, jnp.float64))


def check_germination_step(state, day_number, crop_first_day,
        actual_root_water_mm, field_capacity_mm, wilting_point_mm,
        required_available_water_pct, *, seed_planted=True):
    """CheckGermination including BUDGET's enclosing guard.

    Inputs are full minimum-root-zone water after infiltration, capillary
    rise and salt transport, not the wetter topsoil selected for canopy.
    Equality germinates. Dry days preserve the protection flag, increment
    delay, and reset only sum_gdd. Once germinated the state is unchanged.
    """
    actual, fc, wp, percentage = [jnp.asarray(v, jnp.float64) for v in
        (actual_root_water_mm, field_capacity_mm, wilting_point_mm, required_available_water_pct)]
    threshold = wp + (fc - wp) * (percentage / 100.)
    active = (~jnp.asarray(state.germinated, jnp.bool_) &
        (jnp.asarray(day_number, jnp.int32) >= jnp.asarray(crop_first_day, jnp.int32)))
    dry = active & (actual < threshold)
    wet = active & ~dry
    return CropClockState(jnp.asarray(state.germinated, jnp.bool_) | wet,
        jnp.where(wet, jnp.asarray(seed_planted, jnp.bool_), state.protected_seedling),
        jnp.asarray(state.delayed_days, jnp.int32) + dry.astype(jnp.int32),
        jnp.where(dry, 0., jnp.asarray(state.sum_gdd, jnp.float64)),
        jnp.asarray(state.sum_gdd_from_day1, jnp.float64))
