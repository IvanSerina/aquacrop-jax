"""Biomass, harvest-index and yield operators for the JAX port."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .salinity import round_fortran

F64 = jnp.float64
CO2_REFERENCE = jnp.asarray(369.41, F64)


class ProductionTrajectory(NamedTuple):
    biomass: jax.Array
    unlimited_biomass: jax.Array
    potential_biomass: jax.Array
    harvest_index_pct: jax.Array
    yield_mass: jax.Array
    water_productivity: jax.Array
    biomass_increment: jax.Array


class FertilityStressUpdate(NamedTuple):
    """Daily fertility stress and next-step state from DetermineBiomassAndYield."""
    stress_pct: jax.Array
    previous_stress_pct: jax.Array
    sum_kc_top_stress: jax.Array


def dynamic_fertility_stress_step(
        biomass_total, biomass_unlimited, potential_biomass_fraction,
        coefficient0, coefficient1, coefficient2, management_stress_pct,
        previous_stress_pct, sum_kc_top, *, calibrated=True,
        grain_crop=False, determinacy_linked=False, dayi=0,
        delayed_days=0, crop_day1=1, flowering_day=0,
        flowering_length=0):
    """Port the dynamic soil-fertility update in ``simul.f90``.

    Biomass arguments are cumulative values *after* the current day's biomass
    update, matching ``DetermineBiomassAndYield``. The returned previous
    stress is the value to carry to the next step; ``BUDGET_module`` consumes
    this value before the next production update.
    """
    total, unlimited, fraction, b0, b1, b2, top = [
        jnp.asarray(value, F64) for value in
        (biomass_total, biomass_unlimited, potential_biomass_fraction,
         coefficient0, coefficient1, coefficient2, sum_kc_top)]
    management = jnp.asarray(management_stress_pct, jnp.int32)
    previous = jnp.asarray(previous_stress_pct, jnp.int32)
    calibrated = jnp.asarray(calibrated, jnp.bool_)

    safe_unlimited = jnp.where(unlimited > 0.001, unlimited, 1.0)
    biomass_adjustment = 100.0 * (fraction + (fraction - total / safe_unlimited))
    polynomial = b0 + b1 * biomass_adjustment + b2 * biomass_adjustment ** 2
    rounded = jnp.asarray(round_fortran(polynomial, bits=8), jnp.int32)
    calibrated_stress = jnp.where(
        biomass_adjustment >= 100.0, 0,
        jnp.where(biomass_adjustment <= jnp.finfo(F64).eps, 80,
                  jnp.where(rounded < 0, management,
                            jnp.minimum(rounded, 80))))
    calibrated_stress = jnp.minimum(calibrated_stress, management)

    active = (management > 0) & (unlimited > 0.001) & calibrated
    fallback = jnp.where((management == 0) | (~calibrated), 0, management)
    stress = jnp.where(active, calibrated_stress, fallback)

    half_flowering = jnp.asarray(
        round_fortran(jnp.asarray(flowering_length, F64) / 2.0), jnp.int32)
    flowering_start = (jnp.asarray(delayed_days, jnp.int32) +
                       jnp.asarray(crop_day1, jnp.int32) +
                       jnp.asarray(flowering_day, jnp.int32))
    # Native tmax1 stays undef_int (-9) until the yield block executes,
    # strictly after flowering starts. Its later guard also uses that sentinel.
    tmax1 = jnp.where(jnp.asarray(dayi, jnp.int32) > flowering_start,
                     half_flowering, -9)
    post_vegetative = (jnp.asarray(grain_crop, jnp.bool_) &
        jnp.asarray(determinacy_linked, jnp.bool_) &
        (jnp.asarray(dayi, jnp.int32) > flowering_start + tmax1))
    monotonic = active & post_vegetative
    stress = jnp.where(monotonic, jnp.maximum(stress, previous), stress)
    stress = jnp.where(monotonic, jnp.minimum(stress, management), stress)
    stress = jnp.asarray(stress, jnp.int32)
    return FertilityStressUpdate(
        stress, stress, (1.0 - stress.astype(F64) / 100.0) * top)


def co2_water_productivity_factor(co2_ppm, water_productivity_g_m2,
                                  sink_strength_pct):
    co2 = jnp.asarray(co2_ppm, F64)
    wp = jnp.asarray(water_productivity_g_m2, F64)
    crop_type = jnp.where(wp >= 40.0, 0.0,
                          jnp.where(wp <= 20.0, 1.0, (40.0 - wp) / 20.0))
    sink = jnp.clip(jnp.asarray(sink_strength_pct, F64) / 100.0, 0.0, 1.0)
    weight = jnp.where(co2 <= CO2_REFERENCE, 0.0,
                       jnp.clip((co2 - CO2_REFERENCE) /
                                (550.0 - CO2_REFERENCE), 0.0, 1.0))
    old = (co2 / CO2_REFERENCE) / (
        1.0 + (co2 - CO2_REFERENCE) *
        ((1.0 - weight) * 0.000138 + weight *
         (0.000138 * sink + 0.001165 * (1.0 - sink))))
    shape = -4.61824 - 3.43831 * sink - 5.32587 * sink * sink
    relative = (co2 - CO2_REFERENCE) / (2000.0 - CO2_REFERENCE)
    adjusted = jnp.where(
        co2 >= 2000.0, 1.58,
        1.0 + 0.58 * jnp.expm1(relative * shape) / jnp.expm1(shape))
    selected = jnp.where(co2 <= CO2_REFERENCE, old,
                         jnp.where(jnp.logical_and(co2 <= 550.0, old < adjusted),
                                   old, adjusted))
    return 1.0 + crop_type * (selected - 1.0)


def daily_water_productivity(
        base_water_productivity_g_m2, co2_ppm, sink_strength_pct,
        day_after_planting, flowering_day, reference_harvest_index_pct,
        harvest_index_rate_pct_day, yield_formation_wp_pct=100.0,
        reproductive_crop=True):
    wp_parameter = jnp.asarray(base_water_productivity_g_m2, F64)
    wp = wp_parameter / 100.0
    day = jnp.asarray(day_after_planting, F64)
    flowering = jnp.asarray(flowering_day, F64)
    hi = jnp.asarray(reference_harvest_index_pct, F64)
    rate = jnp.asarray(harvest_index_rate_pct_day, F64)
    formation_days = round_fortran(jnp.where(rate != 0.0, hi / rate, 0.0))
    after_flowering = day - flowering
    third = formation_days / 3.0
    switch = jnp.where(jnp.logical_and(formation_days > 0.0,
                                       after_flowering < third),
                       jnp.where(third != 0.0, after_flowering / third, 0.0), 1.0)
    switch = jnp.clip(switch, 0.0, 1.0)
    reproductive = jnp.logical_and(
        jnp.asarray(reproductive_crop, jnp.bool_),
        jnp.logical_and(day >= flowering,
                        jnp.asarray(yield_formation_wp_pct, F64) < 100.0))
    wp *= jnp.where(
        reproductive,
        1.0 - (1.0 - jnp.asarray(yield_formation_wp_pct, F64) / 100.0) * switch,
        1.0)
    return wp * co2_water_productivity_factor(
        co2_ppm, wp_parameter, sink_strength_pct)


def harvest_index_growth_coefficient(maximum_hi_pct, daily_rate_pct):
    maximum = jnp.asarray(maximum_hi_pct, F64)
    rate = jnp.asarray(daily_rate_pct, F64)
    duration = jnp.where(rate != 0.0, maximum / rate, 0.0)

    def hi_at(coefficient):
        return maximum / (1.0 + (maximum - 1.0) * jnp.exp(-coefficient * duration))

    _, coefficient = lax.while_loop(
        lambda state: jnp.logical_and(hi_at(state[1]) <= 0.98 * maximum,
                                      state[0] < 100000),
        lambda state: (state[0] + 1, state[1] + 0.001),
        (jnp.asarray(0, jnp.int32), jnp.asarray(0.002, F64)))
    return jnp.where(maximum > 1.0, coefficient, jnp.asarray(-9.0, F64))


def reference_harvest_index(day_after_planting, flowering_day,
                            maximum_hi_pct, daily_rate_pct,
                            grain_crop=True):
    """Healthy-canopy reference HI; day is zero on Crop.Day1, as in Fortran.

    Delayed germination, regrowth overrides and the recurrent canopy-driven
    HIfinal restriction belong to the caller and are not represented here.
    """
    day = jnp.asarray(day_after_planting, F64) - jnp.asarray(flowering_day, F64)
    maximum = jnp.asarray(maximum_hi_pct, F64)
    rate = jnp.asarray(daily_rate_pct, F64)

    def immediate(_):
        return jnp.where(day > 0.0, maximum, 0.0)

    def developed(_):
        coefficient = harvest_index_growth_coefficient(maximum, rate)
        maximum_day = jnp.asarray(round_fortran(maximum / rate), jnp.int32)

        def condition(state):
            index, _, _, done = state
            return jnp.logical_and(jnp.logical_not(done),
                                   index <= jnp.maximum(maximum_day, 1))

        def body(state):
            index, previous, switch, done = state
            time = jnp.asarray(index, F64)
            current = maximum / (1.0 + (maximum - 1.0) * jnp.exp(-coefficient * time))
            final = current + jnp.asarray(maximum_day - index, F64) * (current - previous)
            stop = jnp.logical_or(final > maximum, index >= maximum_day)
            candidate = jnp.where(jnp.logical_and(stop, jnp.logical_not(done)),
                                  index - 1, switch)
            return index + 1, current, candidate, jnp.logical_or(done, stop)

        _, _, switch, _ = lax.while_loop(
            condition, body,
            (jnp.asarray(1, jnp.int32), jnp.asarray(1.0, F64),
             jnp.asarray(0, jnp.int32), jnp.asarray(False)))
        switch_f = jnp.asarray(switch, F64)
        at_switch = maximum / (1.0 + (maximum - 1.0) * jnp.exp(-coefficient * switch_f))
        tail_days = jnp.asarray(maximum_day - switch, F64)
        linear_rate = jnp.where(tail_days != 0.0,
                                (maximum - at_switch) / tail_days, 0.0)
        logistic = maximum / (1.0 + (maximum - 1.0) * jnp.exp(-coefficient * day))
        grain_tail = at_switch + linear_rate * (day - switch_f)
        after_switch = jnp.where(jnp.asarray(grain_crop, jnp.bool_), grain_tail, logistic)
        value = jnp.where(day < switch_f, logistic, after_switch)
        # HarvestIndexDay applies this early plateau only to the logistic tail.
        logistic_tail = jnp.logical_and(jnp.logical_not(jnp.asarray(grain_crop, jnp.bool_)),
                                       day >= switch_f)
        value = jnp.where(jnp.logical_and(logistic_tail, value >= 0.9799 * maximum),
                          maximum, value)
        value = jnp.where(value <= 1.4, 0.0, value)
        value = jnp.where(maximum - value < 0.4, maximum, value)
        return jnp.where(day > 0.0, jnp.minimum(value, maximum), 0.0)

    return lax.cond(rate > 99.0, immediate, developed, operand=None)


def run_production_trajectory(
        actual_transpiration, potential_transpiration, eto, co2_ppm,
        day_after_planting, base_water_productivity_g_m2, sink_strength_pct,
        flowering_day, maximum_hi_pct, daily_hi_rate_pct,
        yield_formation_wp_pct=100.0, fertility_wp_reduction_pct=0.0,
        potential_biomass_fraction=1.0, reproductive_crop=True,
        grain_crop=True, initial_biomass=0.0,
        initial_unlimited_biomass=0.0, initial_yield=0.0):
    tact = jnp.asarray(actual_transpiration, F64)
    tpot = jnp.asarray(potential_transpiration, F64)
    eto = jnp.asarray(eto, F64)
    co2 = jnp.asarray(co2_ppm, F64)
    days = jnp.asarray(day_after_planting, F64)
    reduction = 1.0 - jnp.asarray(fertility_wp_reduction_pct, F64) / 100.0
    potential_fraction = jnp.asarray(potential_biomass_fraction, F64)
    reproductive = jnp.asarray(reproductive_crop, jnp.bool_)

    def body(carry, forcing):
        biomass, unlimited, yield_mass = carry
        actual, potential_t, eto_day, co2_day, day = forcing
        wp = daily_water_productivity(
            base_water_productivity_g_m2, co2_day, sink_strength_pct,
            day, flowering_day, maximum_hi_pct, daily_hi_rate_pct,
            yield_formation_wp_pct, reproductive_crop)
        increment = jnp.where(eto_day > 0.0, wp * reduction * actual / eto_day, 0.0)
        potential_increment = jnp.where(eto_day > 0.0, wp * potential_t / eto_day, 0.0)
        biomass += increment
        unlimited += potential_increment
        potential_biomass = potential_fraction * unlimited
        harvest_index = reference_harvest_index(
            day, flowering_day, maximum_hi_pct, daily_hi_rate_pct, grain_crop)
        yield_mass = jnp.where(reproductive, biomass * harvest_index / 100.0,
                               yield_mass + increment * harvest_index / 100.0)
        output = (biomass, unlimited, potential_biomass, harvest_index,
                  yield_mass, wp, increment)
        return (biomass, unlimited, yield_mass), output

    _, output = lax.scan(
        body,
        (jnp.asarray(initial_biomass, F64),
         jnp.asarray(initial_unlimited_biomass, F64),
         jnp.asarray(initial_yield, F64)),
        (tact, tpot, eto, co2, days))
    return ProductionTrajectory(*output)
