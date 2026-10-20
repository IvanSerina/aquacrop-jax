"""Explicit daily recurrence for global.f90::SeasonalSumOfKcPot."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .canopy import (calculate_potential_et, potential_canopy_calendar,
                     potential_canopy_gdd)
from .rounding import round_fortran

F64 = jnp.float64


class ReferenceETTrajectory(NamedTuple):
    canopy: jax.Array
    transpiration_mm: jax.Array
    cumulative_kc: jax.Array
    sum_kc: jax.Array


def reference_crop_transpiration(
        daily_gdd, calendar_days, thermal_days, initial_cover, maximum_cover,
        calendar_growth, thermal_growth, calendar_decline, thermal_decline,
        kc_top, kc_decline_pct, late_evaporation_pct, co2_ppm, minimum_gdd,
        *, gdd_mode=True, initial_cover_day=0, initial_cover_gdd=0):
    """Reference Kc using ETo=5, including first growth day and regrowth.

    Both milestone tuples contain emergence, full canopy, senescence and
    maturity. The climate reader supplies already aligned/cycled daily GDD;
    its static length replaces Lend. Calendar ET uses native DayCC directly,
    even for GDD crops, rather than the simulation's ET calendar conversion.
    """
    daily_gdd = jnp.asarray(daily_gdd, F64)
    if daily_gdd.ndim != 1:
        raise ValueError("Reference climate must be a one-dimensional GDD series")
    if len(calendar_days) != 4 or len(thermal_days) != 4:
        raise ValueError("Reference calendars require four milestones")
    l0, l12, l123, maturity = [jnp.asarray(v, F64) for v in calendar_days]
    g0, g12, g123, g_maturity = [jnp.asarray(v, F64) for v in thermal_days]
    mode = jnp.asarray(gdd_mode, jnp.bool_)
    initial, maximum = jnp.asarray(initial_cover, F64), jnp.asarray(maximum_cover, F64)
    ini_day, ini_gdd = jnp.asarray(initial_cover_day, F64), jnp.asarray(initial_cover_gdd, F64)
    regrowth = ini_day != 0.
    tadj = jnp.where(ini_day == -9., l12 - l0, ini_day)
    gdd_tadj = jnp.where(ini_day == -9., g12 - g0, ini_gdd)
    start_gdd = jnp.where(regrowth, g0 + gdd_tadj, 0.)

    def canopy(day, gdd):
        return lax.cond(mode,
            lambda _: potential_canopy_gdd(gdd, g0, g123, g_maturity,
                initial, maximum, thermal_growth, thermal_decline),
            lambda _: potential_canopy_calendar(day, l0, l123, maturity,
                initial, maximum, calendar_growth, calendar_decline), None)

    initial_cc = jnp.where(ini_day == -9., maximum,
        jnp.where(regrowth, canopy(tadj + l0, start_gdd), initial))
    day_fraction = (l123 - l12) / jnp.where(regrowth, tadj + l0 + l123 - l12, 1.)
    gdd_fraction = (g123 - g12) / jnp.where(regrowth & mode,
        gdd_tadj + g0 + g123 - g12, 1.)

    def step(state, forcing):
        total_gdd, from_day1, growth_on, sum_kc = state
        day, daily = forcing
        total_gdd = jnp.where(mode, total_gdd + daily, total_gdd)
        from_day1 = jnp.where(mode, from_day1 + daily, from_day1)

        def first(_):
            starts = regrowth | jnp.where(mode, total_gdd > g0, day == l0 + 1.)
            return jnp.where(starts, initial_cc, 0.), day, starts

        def developed(_):
            adjusted_day = jnp.minimum(day + tadj + l0, maturity)
            adjusted_day = jnp.where(adjusted_day > l12,
                jnp.where(day <= l123,
                    l12 + round_fortran(day_fraction * (day + tadj + l0 - l12)),
                    day), adjusted_day)
            day_cc = jnp.where(regrowth, adjusted_day, day)
            plot_gdd = jnp.minimum(total_gdd, g_maturity)
            plot_gdd = jnp.where(plot_gdd > g12,
                jnp.where(from_day1 <= g123,
                    g12 + round_fortran(gdd_fraction * (from_day1 + gdd_tadj + g0 - g12)),
                    from_day1), plot_gdd)
            plot_gdd = jnp.where(ini_gdd == 0., from_day1, plot_gdd)
            return canopy(day_cc, plot_gdd), day_cc, jnp.bool_(True)

        cc, day_cc, growth_on = lax.cond(growth_on, developed, first, None)
        # Native overwrites this with today's CC, not its historical maximum.
        withered = jnp.where(day >= l12, maximum, cc)
        demand = calculate_potential_et(day_cc, l0, l12, l123, maturity,
            cc, 5., kc_top, kc_decline_pct, maximum, withered,
            late_evaporation_pct, co2_ppm, daily, minimum_gdd)
        tpot = jnp.where(cc > .0001, demand.transpiration_mm, 0.)
        sum_kc = sum_kc + tpot / 5.
        return (total_gdd, from_day1, growth_on, sum_kc), (cc, tpot, sum_kc)

    initial_state = (jnp.where(mode, start_gdd, -9.), jnp.asarray(0., F64),
                     jnp.bool_(False), jnp.asarray(0., F64))
    final, (cover, tpot, cumulative) = lax.scan(step, initial_state,
        (jnp.arange(1, daily_gdd.shape[0] + 1, dtype=F64), daily_gdd))
    return ReferenceETTrajectory(cover, tpot, cumulative, final[3])
