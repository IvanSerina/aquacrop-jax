"""Actual-climate coefficient conversion, separate from reference climate."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .rounding import round_fortran
from .thermal_time import calendar_days_from_record


class CalendarDeclineResult(NamedTuple):
    coefficient: jax.Array
    decline_gdd: jax.Array
    decline_days: jax.Array
    valid: jax.Array


def calendar_decline_from_record(daily_gdd, maximum_cover, thermal_decline,
        senescence_gdd, maturity_gdd, senescence_day, first_offset=0, *,
        undefined_year=False):
    """GDDCDCToCDC with Reference=False on a full daily temperature record.

    The temperature lookup starts at planting plus calendar senescence.
    For senescence at/after harvest, preserve the native 5-GDD fallback.
    A zero/invalid canopy or unusable climate is signaled via valid=False.
    """
    maximum, cdc = [jnp.asarray(v, jnp.float64) for v in (maximum_cover, thermal_decline)]
    sen, maturity, day, offset = [jnp.asarray(v, jnp.int32) for v in
        (senescence_gdd, maturity_gdd, senescence_day, first_offset)]
    positive = cdc > jnp.finfo(jnp.float64).eps
    full = jnp.where(maximum > 0., jnp.where(positive,
        round_fortran(((maximum + 2.29) / (jnp.where(positive, cdc, 1.) * 3.33)) *
                      jnp.log(21.) + .50), -9.), 0.).astype(jnp.int32)
    complete = sen + full <= maturity
    heat = jnp.where(complete, full, jnp.where(sen < maturity, maturity - sen, 5))
    cover = jnp.where(complete, 0., maximum * (1. - .05 * (
        jnp.exp(heat.astype(jnp.float64) * (cdc * 3.33) / (maximum + 2.29)) - 1.)))
    duration = calendar_days_from_record(daily_gdd, heat, offset + day,
                                         undefined_year=undefined_year)
    coefficient = (((maximum + 2.29) / jnp.where(duration.value > 0, duration.value, 1)) *
        jnp.log(1. + ((1. - cover / jnp.where(maximum != 0., maximum, 1.)) / .05))) / 3.33
    valid = (maximum > 0.) & duration.valid & (duration.value > 0) & jnp.isfinite(coefficient)
    return CalendarDeclineResult(jnp.where(valid, coefficient, -9.), heat,
                                  duration.value, valid)
