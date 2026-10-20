"""Actual-calendar daily fertility forcing before the water/canopy budget."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .fertility import FertilityEffect, soil_fertility_effect, time_to_max_canopy_sf
from .thermal_time import growing_degree_days_from_record


class DailyFertilityParameters(NamedTuple):
    calendar_days: tuple
    thermal_days: tuple
    initial_cover: jax.Array
    maximum_cover: jax.Array
    calendar_growth: jax.Array
    flowering_day: jax.Array
    flowering_length: jax.Array
    determinacy_linked: jax.Array
    response_shapes: tuple


class DailyFertilityResponse(NamedTuple):
    effect: FertilityEffect
    full_canopy_day: jax.Array
    full_canopy_gdd: jax.Array
    stress_pct: jax.Array
    valid: jax.Array


def daily_fertility_no_salinity_step(stress_pct, virtual_time, germinated,
        management_stress_pct, last_virtual_day, parameters, record_daily_gdd,
        first_offset, *, undefined_year=False):
    """EffectSoilFertilitySalinityStress with SalinityConsidered=False.

    Calendar/GDD parameters are the loaded actual crop, not reference-year
    calibration parameters. GDD forcing is the complete daily temperature
    file record, with sowing offset already normalized by the caller.
    The no-stress/germination guards restore BASE full-canopy milestones.
    In the active branch, only nonzero management triggers GrowingDegreeDays;
    management zero keeps the base GDD even when calendar timing is adjusted.
    Constant-temperature and external-array native branches are excluded.
    """
    p = parameters
    stress, virtual, management, last = [jnp.asarray(v, jnp.int32) for v in
        (stress_pct, virtual_time, management_stress_pct, last_virtual_day)]
    active = ((virtual >= p.calendar_days[0]) & (virtual <= last)
              & jnp.asarray(germinated, jnp.bool_) & (stress != 0))

    def no_effect(_):
        zero = jnp.asarray(0., jnp.float64)
        return DailyFertilityResponse(FertilityEffect(zero, zero, zero, zero, zero),
            jnp.asarray(p.calendar_days[1], jnp.int32),
            jnp.asarray(p.thermal_days[1], jnp.int32), stress, jnp.bool_(True))

    def adjusted(_):
        effect = soil_fertility_effect(stress, *p.response_shapes)
        timing = time_to_max_canopy_sf(p.initial_cover, p.calendar_growth,
            p.maximum_cover, p.calendar_days[0], p.calendar_days[1], p.calendar_days[2],
            p.flowering_day, p.flowering_length, p.determinacy_linked,
            effect.reduction_growth_pct, effect.reduction_cover_pct, stress)
        effect = effect._replace(
            reduction_growth_pct=timing.reduction_growth_pct.astype(jnp.float64),
            reduction_cover_pct=timing.reduction_cover_pct.astype(jnp.float64))
        def thermal(_):
            converted = growing_degree_days_from_record(record_daily_gdd,
                timing.full_canopy_day, first_offset, undefined_year=undefined_year)
            return converted.value, converted.valid
        gdd, valid = lax.cond(management != 0, thermal,
            lambda _: (jnp.asarray(p.thermal_days[1], jnp.int32), jnp.bool_(True)), None)
        return DailyFertilityResponse(effect, timing.full_canopy_day, gdd,
                                      timing.stress_class_pct, valid)

    return lax.cond(active, adjusted, no_effect, None)
