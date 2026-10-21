"""Ordered seeded GDD clock, germination and canopy state for a daily driver."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .canopy_step import CanopyDayState, CanopyDayDiagnostics, canopy_gdd_step
from .crop_clock import (CropClockState, accumulate_crop_gdd,
                         check_germination_step, seeded_canopy_clock)


class SeededCropState(NamedTuple):
    germinated: jax.Array
    delayed_days: jax.Array
    sum_gdd: jax.Array
    sum_gdd_from_day1: jax.Array
    canopy: CanopyDayState

    @property
    def clock(self):
        # Native Simulation.ProtectedSeedling is one shared flag. The canopy
        # owns it here; there is no second cached copy to revive protection.
        return CropClockState(self.germinated, self.canopy.canopy.protected,
            self.delayed_days, self.sum_gdd, self.sum_gdd_from_day1)


def _with_clock(state, clock):
    canopy = state.canopy._replace(canopy=state.canopy.canopy._replace(
        protected=clock.protected_seedling))
    return SeededCropState(clock.germinated, clock.delayed_days,
        clock.sum_gdd, clock.sum_gdd_from_day1, canopy)


def begin_seeded_crop_day(state, day_number, daily_gdd, crop_first_day,
        simulation_first_day, simulation_last_day, emergence_day, initial_cover):
    """Capture the seeded AdvanceOneTimeStep clock before the water budget.

    Daily heat is accumulated first, using SetGDDVariablesNextDay semantics.
    This path assumes the simulation started before or on planting; mid-crop
    initialization and regrowth require their separate native preparations.
    """
    clock = accumulate_crop_gdd(state.clock, daily_gdd, day_number,
                                crop_first_day, simulation_last_day)
    captured = seeded_canopy_clock(clock, day_number, crop_first_day)
    state = _with_clock(state, clock)
    day, first, start, emergence = [jnp.asarray(v, jnp.int32) for v in
        (day_number, crop_first_day, simulation_first_day, emergence_day)]
    initialize_previous = (day == first + emergence) & (day > start)
    previous = jnp.where(initialize_previous, jnp.asarray(initial_cover, jnp.float64),
                         state.canopy.canopy.cover)
    return state._replace(canopy=state.canopy._replace(
        canopy=state.canopy.canopy._replace(cover=previous))), captured


def germinate_seeded_crop_step(state, day_number, crop_first_day,
        actual_root_water_mm, field_capacity_mm, wilting_point_mm,
        required_available_water_pct, *, seed_planted=True):
    """BUDGET germination after water/salt inputs, before daily SF and canopy."""
    clock = check_germination_step(state.clock, day_number, crop_first_day,
        actual_root_water_mm, field_capacity_mm, wilting_point_mm,
        required_available_water_pct, seed_planted=seed_planted)
    return _with_clock(state, clock)


def seeded_canopy_step(state, captured_clock, daily_gdd, eto_mm,
        effective_water_mm, effective_fc_mm, effective_wp_mm, parameters, effect):
    """Canopy after germination/SF, keeping the pre-budget local GDD clock.

    Fertility response and SF milestones must be computed using the updated
    germination state, as BUDGET does between germination and canopy.
    The calendar expression deliberately uses the post-check delayed_days.
    """
    calendar_day = captured_clock.virtual_day + state.delayed_days + 1
    def stopped(_):
        unset = jnp.float64(-9.)
        return state.canopy, CanopyDayDiagnostics(unset, unset, unset, unset, jnp.bool_(True))

    canopy, diagnostic = lax.cond(state.canopy.senescence.no_more_crop, stopped,
        lambda _: canopy_gdd_step(state.canopy, captured_clock.cumulative_gdd,
            daily_gdd, calendar_day, eto_mm, effective_water_mm, effective_fc_mm,
            effective_wp_mm, parameters, effect), None)
    return state._replace(canopy=canopy), diagnostic
