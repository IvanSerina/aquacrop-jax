"""Potential canopy-cover and evapotranspiration operators."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

F64 = jnp.float64


class PotentialET(NamedTuple):
    transpiration_mm: jax.Array
    evaporation_mm: jax.Array
    adjusted_canopy: jax.Array
    crop_coefficient: jax.Array
    cold_stress: jax.Array


def canopy_at_time(time, initial_cover, growth_coefficient, maximum_cover):
    """Translate ``CCatTime``/``CCatGDD`` (same continuous equation)."""
    time, initial, growth, maximum = [jnp.asarray(value, F64) for value in
                                      (time, initial_cover,
                                       growth_coefficient, maximum_cover)]
    exponential = initial * jnp.exp(growth * time)
    logistic = maximum - 0.25 * (maximum / initial) * maximum * jnp.exp(
        -growth * time)
    return jnp.where(exponential > maximum / 2.0, logistic, exponential)


def potential_canopy_gdd(cumulative_gdd, emergence_gdd, senescence_gdd,
                         maturity_gdd, initial_cover, maximum_cover,
                         growth_coefficient, decline_coefficient,
                         fertility_growth_reduction_pct=0.0,
                         fertility_cover_reduction_pct=0.0):
    """Exact no-water-stress GDD path of ``CanopyCoverNoStressSF``."""
    gdd, emergence, senescence, maturity, initial, maximum, growth, decline = [
        jnp.asarray(value, F64) for value in
        (cumulative_gdd, emergence_gdd, senescence_gdd, maturity_gdd,
         initial_cover, maximum_cover, growth_coefficient,
         decline_coefficient)]
    growth *= 1.0 - jnp.asarray(fertility_growth_reduction_pct, F64) / 100.0
    adjusted_maximum = maximum * (
        1.0 - jnp.asarray(fertility_cover_reduction_pct, F64) / 100.0)
    growing = canopy_at_time(gdd - emergence, initial, growth,
                             adjusted_maximum)
    cover_at_senescence = canopy_at_time(
        senescence - emergence, initial, growth, adjusted_maximum)
    adjusted_decline = decline * (cover_at_senescence + 2.29) / (
        maximum + 2.29)
    declining = cover_at_senescence * (
        1.0 - 0.05 * jnp.expm1(
            (gdd - senescence) * 3.33 * adjusted_decline /
            (cover_at_senescence + 2.29)))
    # roundc uses ties-to-even; salinity imports canopy, so keep this primitive
    # here rather than introducing a circular import through round_fortran.
    active = jnp.logical_and(gdd > 0.0, jnp.rint(gdd) <= maturity)
    cover = jnp.where(gdd <= emergence, 0.0,
                      jnp.where(gdd < senescence, growing, declining))
    valid = jnp.logical_and(active, initial > 0.0)
    valid = jnp.logical_and(valid, jnp.logical_or(
        gdd < senescence, jnp.logical_and(
            maximum >= 0.001, cover_at_senescence >= 0.001)))
    return jnp.where(valid, jnp.clip(cover, 0.0, 1.0), 0.0)


def potential_canopy_calendar(day_after_planting, emergence_day,
                              senescence_day, maturity_day, initial_cover,
                              maximum_cover, growth_coefficient,
                              decline_coefficient, delayed_days=0,
                              fertility_growth_reduction_pct=0.0,
                              fertility_cover_reduction_pct=0.0):
    """Calendar-day path of ``CanopyCoverNoStressSF``."""
    day = jnp.asarray(day_after_planting, F64) - jnp.asarray(delayed_days, F64)
    emergence = jnp.asarray(emergence_day, F64)
    senescence = jnp.asarray(senescence_day, F64)
    maturity = jnp.asarray(maturity_day, F64)
    initial = jnp.asarray(initial_cover, F64)
    maximum = jnp.asarray(maximum_cover, F64)
    growth = jnp.asarray(growth_coefficient, F64) * (
        1.0 - jnp.asarray(fertility_growth_reduction_pct, F64) / 100.0)
    adjusted_maximum = maximum * (
        1.0 - jnp.asarray(fertility_cover_reduction_pct, F64) / 100.0)
    growing = canopy_at_time(day - emergence, initial, growth,
                             adjusted_maximum)
    cover_at_senescence = canopy_at_time(
        senescence - emergence, initial, growth, adjusted_maximum)
    adjusted_decline = jnp.asarray(decline_coefficient, F64) * (
        cover_at_senescence + 2.29) / (maximum + 2.29)
    declining = cover_at_senescence * (
        1.0 - 0.05 * jnp.expm1(
            (day - senescence) * 3.33 * adjusted_decline /
            (cover_at_senescence + 2.29)))
    cover = jnp.where(day <= emergence, 0.0,
                      jnp.where(day < senescence, growing, declining))
    active = jnp.logical_and(day >= 1.0, day <= maturity)
    valid = jnp.logical_and(active, initial > jnp.finfo(F64).eps)
    # Unlike the GDD routine, DaysSF spells this literal without _dp.
    threshold = jnp.asarray(jnp.float32(0.001), F64)
    valid = jnp.logical_and(valid, jnp.logical_or(
        day < senescence, jnp.logical_and(
            maximum >= threshold, cover_at_senescence >= threshold)))
    # Calendar mode removes sub-epsilon cover; the GDD routine only clips < 0.
    cover = jnp.where(cover < jnp.finfo(F64).eps, 0.0,
                      jnp.minimum(cover, 1.0))
    return jnp.where(valid, cover, 0.0)


def temperature_stress(lower_temperature, upper_temperature, temperature):
    """Translate the logistic ``KsTemperature`` response."""
    lower, upper, value = [jnp.asarray(item, F64) for item in
                           (lower_temperature, upper_temperature, temperature)]
    relative = jnp.where(upper != lower, (value - lower) / (upper - lower), 0.0)
    minimum = jnp.asarray(0.02, F64)
    rate = -jnp.log((minimum - 0.98 * minimum) /
                        (0.98 * (1.0 - minimum)))
    coefficient = minimum / (minimum + (1.0 - minimum) *
                             jnp.exp(-rate * relative))
    coefficient -= minimum * (1.0 - relative)
    increasing = lower < upper
    bounded_increasing = jnp.where(value <= lower, 0.0,
                                   jnp.where(value >= upper, 1.0, coefficient))
    bounded_decreasing = jnp.where(value >= lower, 0.0,
                                   jnp.where(value <= upper, 1.0, coefficient))
    response = jnp.where(increasing, bounded_increasing, bounded_decreasing)
    return jnp.where(jnp.abs(lower - upper) <= 1e-12, 1.0,
                     jnp.clip(response, 0.0, 1.0))


def calculate_potential_et(
        day_after_planting, emergence_day, full_canopy_day, senescence_day,
        harvest_day, canopy_cover, eto_mm, maximum_crop_coefficient,
        cumulative_crop_coefficient_decline_pct, maximum_canopy_cover,
        maximum_withered_canopy, late_evaporation_effect_pct, co2_ppm,
        daily_gdd, minimum_gdd_full_transpiration, wet_bare_coefficient=1.10,
        last_cut_day=0, delayed_days=0, senescence_exponent=2.5,
        evaporation_limit=True):
    """Translate ``CalculateETpot`` with explicit simulation parameters."""
    day = jnp.asarray(day_after_planting, F64) - jnp.asarray(delayed_days, F64)
    emergence, full, senescence, harvest, canopy, eto, kc, maximum, withered, bare = [
        jnp.asarray(v, F64) for v in (emergence_day, full_canopy_day,
        senescence_day, harvest_day, canopy_cover, eto_mm,
        maximum_crop_coefficient, maximum_canopy_cover,
        maximum_withered_canopy, wet_bare_coefficient)]
    adjusted = jnp.clip(1.72 * canopy - canopy ** 2 + 0.30 * canopy ** 3, 0.0, 1.0)
    denominator = harvest - full
    relative_time = jnp.where(denominator != 0.0,
                              (day - jnp.asarray(last_cut_day, F64) - full) / denominator, 0.0)
    ageing = jnp.expm1(relative_time) / jnp.expm1(jnp.asarray(1.0, F64))
    aged_kc = kc - ageing * (jnp.asarray(
        cumulative_crop_coefficient_decline_pct, F64) / 100.0) * withered
    kc = jnp.where(day - jnp.asarray(last_cut_day, F64) > full, aged_kc, kc)
    co2 = jnp.asarray(co2_ppm, F64)
    kc *= jnp.where(co2 > 369.41,
                   1.0 - 0.05 * (co2 - 369.41) / (550.0 - 369.41), 1.0)
    gdd = jnp.asarray(daily_gdd, F64)
    cold = jnp.where(jnp.logical_or(adjusted <= 1e-12, jnp.floor(gdd + 0.5) < 0.0),
                    1.0, temperature_stress(
                        0.0, minimum_gdd_full_transpiration, gdd))
    transpiration = adjusted * cold * kc * eto
    evaporation = bare * (1.0 - adjusted) * eto
    maximum_evaporation = bare * eto * (
        1.0 - withered * jnp.asarray(late_evaporation_effect_pct, F64) / 100.0)
    multiplier = jnp.where(canopy > maximum / 2.0,
                          jnp.where(canopy > maximum, 0.0,
                                   (maximum - canopy) / (maximum / 2.0)), 1.0)
    late_evaporation = evaporation * (
        1.0 - maximum * jnp.asarray(late_evaporation_effect_pct, F64) /
        100.0 * multiplier)
    minimum_evaporation = jnp.maximum(
        bare * (1.0 - 1.72 * maximum + maximum ** 2 - 0.30 * maximum ** 3) * eto,
        0.0)
    late_evaporation = jnp.clip(
        late_evaporation, minimum_evaporation, maximum_evaporation)
    evaporation = jnp.where(jnp.logical_and(day > senescence, maximum > 1e-12),
                           late_evaporation, evaporation)
    evaporation = jnp.where(jnp.asarray(evaporation_limit, jnp.bool_),
                            jnp.minimum(evaporation, maximum_evaporation), evaporation)
    dying = jnp.logical_and(canopy < withered,
                            jnp.logical_and(withered > 0.01, canopy > 0.001))
    transpiration *= jnp.where(
        dying, jnp.exp(jnp.asarray(senescence_exponent, F64) *
                       jnp.log(canopy / withered)), 1.0)
    outside = jnp.logical_or(
        jnp.logical_and(day < emergence, jnp.floor(100.0 * canopy + 0.5) == 0.0),
        day > harvest)
    transpiration = jnp.where(outside, 0.0, transpiration)
    evaporation = jnp.where(outside, bare * eto, evaporation)
    return PotentialET(transpiration, evaporation, adjusted, kc, cold)
