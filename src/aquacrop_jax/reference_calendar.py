"""GDD-mode reference calendar used before fertility/salinity calibration."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .reference_climate import reference_calendar_days
from .rounding import round_fortran

F64 = jnp.float64


class ReferenceCropCalendar(NamedTuple):
    emergence_day: jax.Array
    full_canopy_day: jax.Array
    flowering_day: jax.Array
    flowering_length: jax.Array
    senescence_day: jax.Array
    maturity_day: jax.Array
    hi_days: jax.Array
    calendar_growth: jax.Array
    calendar_decline: jax.Array
    hi_rate: jax.Array
    valid: jax.Array


def gdd_reference_calendar(daily_gdd, calendar_days, thermal_days,
        thermal_growth, thermal_decline, maximum_cover, reference_hi_pct,
        hi_gdd, *, forage_crop=False, reproductive_crop=False,
        grain_crop=False, initial_cover_day=0, initial_cover_gdd=0):
    """AdjustCalendarDaysReferenceTnx with explicit native-SP daily GDD.

    Milestones contain emergence, full canopy, senescence, maturity, flowering
    and flowering length. For forage, native calendar senescence/maturity
    remain the already loaded season length. Invalid nonterminating climates
    or zero full-canopy periods are signaled; callers must require valid.
    """
    if len(calendar_days) != 6 or len(thermal_days) != 6:
        raise ValueError("Reference calendar requires six calendar/thermal milestones")
    l0, full, sen, maturity, flower, length = [jnp.asarray(v, jnp.int32) for v in calendar_days]
    g0, gfull, gsen, gmaturity, gflower, glength = thermal_days
    emergence = reference_calendar_days(daily_gdd, g0)
    full_new = reference_calendar_days(daily_gdd, gfull)
    extra = reference_calendar_days(daily_gdd, gfull - g0 - initial_cover_gdd)
    initial_day = jnp.asarray(initial_cover_day, jnp.int32)
    l0 = jnp.where(initial_day == 0, emergence.days, l0)
    full = jnp.where(initial_day == 0, full_new.days,
        jnp.where(initial_day > 0, l0 + initial_day + extra.days, full))
    valid = jnp.where(initial_day == 0, emergence.valid & full_new.valid,
                      jnp.where(initial_day > 0, extra.valid, True))
    sen_new = reference_calendar_days(daily_gdd, gsen)
    maturity_new = reference_calendar_days(daily_gdd, gmaturity)
    forage = jnp.asarray(forage_crop, jnp.bool_)
    sen = jnp.where(forage, sen, sen_new.days)
    maturity = jnp.where(forage, maturity, maturity_new.days)
    valid &= forage | (sen_new.valid & maturity_new.valid)
    flower_new = reference_calendar_days(daily_gdd, gflower)
    reproductive = jnp.asarray(reproductive_crop, jnp.bool_)
    flower = jnp.where(reproductive, flower_new.days, flower)
    length_new = reference_calendar_days(daily_gdd, glength, start_offset=flower)
    length = jnp.where(reproductive,
        jnp.where(jnp.asarray(grain_crop), length_new.days, 0), length)
    hi = reference_calendar_days(daily_gdd, hi_gdd,
        start_offset=jnp.where(reproductive, flower, 0))
    valid &= hi.valid & (~reproductive | flower_new.valid)
    valid &= (~(reproductive & jnp.asarray(grain_crop))) | length_new.valid
    growth = (jnp.asarray(gfull, F64) / jnp.where(full > 0, full, 1)) * thermal_growth
    valid &= full > 0

    maximum = jnp.asarray(maximum_cover, F64)
    cdc = jnp.asarray(thermal_decline, F64)
    full_decline = jnp.where(maximum > 0.,
        jnp.where(cdc <= jnp.finfo(F64).eps, -9., round_fortran(
            ((maximum + 2.29) / (jnp.where(cdc > jnp.finfo(F64).eps, cdc, 1.) * 3.33)) *
            jnp.log(21.) + .50)), 0.)
    complete = jnp.asarray(gsen, F64) + full_decline <= jnp.asarray(gmaturity, F64)
    decline_gdd = jnp.where(complete, full_decline,
        jnp.where(jnp.asarray(gsen) < jnp.asarray(gmaturity),
                   jnp.asarray(gmaturity) - jnp.asarray(gsen), 5.))
    remaining_cover = jnp.where(complete, 0., maximum * (1. - .05 * (
        jnp.exp(decline_gdd * (cdc * 3.33) / (maximum + 2.29)) - 1.)))
    # GDDCDCToCDC passes the same reference/start date, so offset=0 here too.
    decline_days = reference_calendar_days(daily_gdd, decline_gdd)
    cal_decline = jnp.where(decline_days.days > 0,
        (((maximum + 2.29) / jnp.where(decline_days.days > 0, decline_days.days, 1)) *
         jnp.log(1. + ((1. - remaining_cover / jnp.where(maximum != 0., maximum, 1.)) / .05))) / 3.33,
        -9.)
    valid &= decline_days.valid
    rate_reproductive = jnp.asarray(reference_hi_pct, F64) / jnp.where(hi.days > 0, hi.days, 1)
    denominator = jnp.minimum(hi.days, maturity)
    rate_other = jnp.where(hi.days > 0,
        jnp.asarray(reference_hi_pct, F64) / jnp.where(denominator > 0, denominator, 1), 100.)
    hi_days = jnp.where(reproductive, hi.days,
        jnp.where((hi.days <= 0) | (rate_other > 100.), 0, hi.days))
    rate_other = jnp.minimum(rate_other, 100.)
    rate = jnp.where(reproductive, rate_reproductive, rate_other)
    valid &= (~reproductive) | (hi.days > 0)
    return ReferenceCropCalendar(l0, full, flower, length, sen, maturity,
        hi_days, growth, cal_decline, rate, valid)
