"""Post-production cutting state from run.f90::AdvanceOneTimeStep."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax


class CuttingState(NamedTuple):
    count: jax.Array
    last_cut_day: jax.Array
    interval_days: jax.Array
    interval_gdd: jax.Array
    previous_biomass: jax.Array
    previous_yield: jax.Array
    potential_withered_cover: jax.Array


class CuttingRecord(NamedTuple):
    occurred: jax.Array
    count: jax.Array
    interval_days: jax.Array
    biomass_harvested: jax.Array
    yield_harvested: jax.Array
    canopy_reduced: jax.Array


class ManualCuttingDecision(NamedTuple):
    next_index: jax.Array
    occurred: jax.Array
    finished: jax.Array


def manual_cutting_schedule_step(index, day_number, scheduled_days, *,
                                 enabled=True, crop_first_day=None):
    """Native manual GetNextHarvest and AdvanceOneTimeStep schedule guard.

    Dates retain file order. At most one record is consumed per call, even
    when several dates are overdue. Supply ``crop_first_day`` only for an
    absolute MAN reference: GetNextHarvest then skips records before the
    growing cycle at initialization and after each harvest. Relative MAN
    records are already resolved against the crop start by the host parser.
    """
    days = jnp.asarray(scheduled_days, jnp.int32)
    index, day = [jnp.asarray(v, jnp.int32) for v in (index, day_number)]
    count = days.shape[0]
    if count == 0:
        return ManualCuttingDecision(index, jnp.bool_(False), jnp.bool_(True))

    def scroll(pointer):
        if crop_first_day is None:
            return pointer
        return lax.while_loop(
            lambda i: (i < count) & (days[jnp.clip(i, 0, count - 1)] < crop_first_day),
            lambda i: i + 1, pointer)

    def considered(_):
        current = scroll(index)
        due = ((current < count) & (current >= 0)
               & (day >= days[jnp.clip(current, 0, count - 1)]))
        next_index = lax.cond(due, lambda _: scroll(current + 1), lambda _: current, None)
        return ManualCuttingDecision(next_index, due, next_index >= count)

    return lax.cond(jnp.asarray(enabled, jnp.bool_), considered,
        lambda _: ManualCuttingDecision(index, jnp.bool_(False), index >= count), None)


def apply_cutting_event(canopy_state, cutting_state, occurred, day_in_season,
        cut_cover_pct, cumulative_biomass, cumulative_yield):
    """Apply a decided harvest after the day's production, before output.

    The caller advances interval counters and evaluates the schedule first.
    This is the native HarvestNow block: it records increments before resetting
    counters. Cumulative biomass/yield and crop GDD are never reset. Today's
    actual cover belongs to the daily diagnostic; the returned canopy cover
    is CCiPrev for the following day. Other canopy/senescence state persists.
    """
    state = CuttingState(*[jnp.asarray(v, jnp.int32) for v in cutting_state[:3]],
                         *[jnp.asarray(v, jnp.float64) for v in cutting_state[3:]])
    occurred = jnp.asarray(occurred, jnp.bool_)
    day = jnp.asarray(day_in_season, jnp.int32)
    # The strict predicate needs the native quotient: multiplying by .01
    # changes the 57% boundary by one ULP under XLA.
    divisor = lax.optimization_barrier(jnp.asarray(100., jnp.float64))
    threshold = jnp.asarray(cut_cover_pct, jnp.int32).astype(jnp.float64) / divisor
    biomass, yield_mass = [jnp.asarray(v, jnp.float64)
                          for v in (cumulative_biomass, cumulative_yield)]

    def harvest(_):
        reduced = canopy_state.canopy.cover > threshold
        next_canopy = canopy_state._replace(
            canopy=canopy_state.canopy._replace(
                cover=jnp.where(reduced, threshold, canopy_state.canopy.cover),
                maximum_adjusted=jnp.where(reduced, threshold,
                                           canopy_state.canopy.maximum_adjusted)),
            senescence=canopy_state.senescence._replace(withered_cover=jnp.where(
                reduced, 0., canopy_state.senescence.withered_cover)))
        next_state = CuttingState(state.count + 1, day, jnp.int32(0),
            jnp.float64(0.), biomass, yield_mass,
            jnp.where(reduced, 0., state.potential_withered_cover))
        record = CuttingRecord(jnp.bool_(True), next_state.count, state.interval_days,
            biomass - state.previous_biomass, yield_mass - state.previous_yield, reduced)
        return next_canopy, next_state, record

    def no_harvest(_):
        return canopy_state, state, CuttingRecord(jnp.bool_(False), state.count,
            state.interval_days, jnp.float64(0.), jnp.float64(0.), jnp.bool_(False))

    return lax.cond(occurred, harvest, no_harvest, None)
