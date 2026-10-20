"""Crop response to soil fertility stress, independent of event state."""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .rounding import round_fortran
from .stress import stress_coefficient

F64 = jnp.float64


class FertilityEffect(NamedTuple):
    reduction_growth_pct: jax.Array
    reduction_cover_pct: jax.Array
    reduction_water_productivity_pct: jax.Array
    canopy_decline: jax.Array
    reduction_stomatal_pct: jax.Array


class FertilityCanopyTiming(NamedTuple):
    full_canopy_day: jax.Array
    reduction_growth_pct: jax.Array
    reduction_cover_pct: jax.Array
    stress_class_pct: jax.Array


def fertility_biomass_fraction(management_stress_pct, coefficient0,
                               coefficient1, coefficient2, *, calibrated=True):
    """Invert the fertility polynomial as in run.f90's setup recurrence.

    The native search descends from 100% biomass and linearly interpolates
    the first bracket. Values outside [0, 1] are intentional legacy results;
    do not replace this recurrence with a quadratic root or clip its output.
    Scalar inputs support jit, vmap and scan with a static scalar carry.
    """
    management, b0, b1, b2 = [jnp.asarray(v, F64) for v in
        (management_stress_pct, coefficient0, coefficient1, coefficient2)]

    def invert(_):
        def step(state):
            low, low_stress, _, _, _ = state
            top, top_stress = low, low_stress
            low = low - 1
            biomass_pct = low.astype(F64)
            low_stress = b0 + b1 * biomass_pct + b2 * biomass_pct * biomass_pct
            done = ((low_stress >= management) | (low <= 0) |
                    (low_stress >= 99.99))
            return low, low_stress, top, top_stress, done

        _, low_stress, top, top_stress, _ = lax.while_loop(
            lambda s: ~s[4], step,
            (jnp.int32(100), jnp.asarray(0., F64), jnp.int32(100),
             jnp.asarray(0., F64), jnp.bool_(False)))
        low_stress = jnp.where(low_stress >= 99.99, 100., low_stress)
        difference = low_stress - top_stress
        flat = jnp.abs(difference) < .001
        safe_difference = jnp.where(flat, 1., difference)
        fraction_pct = jnp.where(flat, top.astype(F64),
            top.astype(F64) - (management - top_stress) / safe_difference)
        return fraction_pct / 100.

    return lax.cond((jnp.abs(management) > jnp.finfo(F64).eps) &
                    jnp.asarray(calibrated, jnp.bool_),
                    invert, lambda _: jnp.asarray(1., F64), None)


def days_to_reach_cover(target_cover, initial_cover, maximum_cover,
                        growth_coefficient, emergence_day):
    """``global.f90::DaysToReachCCwithGivenCGC`` with roundc days."""
    target, initial, maximum, growth = [jnp.asarray(v, F64) for v in
        (target_cover, initial_cover, maximum_cover, growth_coefficient)]
    skip = (initial > target) | (initial >= maximum)
    target = jnp.minimum(target, 0.98 * maximum)
    early = jnp.log(target / initial) / growth
    late = jnp.log((0.25 * maximum * maximum / initial) /
                   (maximum - target)) / growth
    duration = jnp.where(skip, 0.0,
                         jnp.where(target <= maximum / 2.0, early, late))
    return (jnp.asarray(emergence_day, jnp.int32) +
            round_fortran(duration).astype(jnp.int32))


def time_to_max_canopy_sf(initial_cover, growth_coefficient, maximum_cover,
                          emergence_day, full_canopy_day, senescence_day,
                          flowering_day, flowering_length, determinant_crop,
                          reduction_growth_pct, reduction_cover_pct,
                          stress_class_pct):
    """``global.f90::TimeToMaxCanopySF`` including both limiting loops."""
    initial, growth, maximum = [jnp.asarray(v, F64) for v in
                                (initial_cover, growth_coefficient, maximum_cover)]
    red_growth = jnp.asarray(reduction_growth_pct, jnp.int32)
    red_cover = jnp.asarray(reduction_cover_pct, jnp.int32)
    stress_class = jnp.asarray(stress_class_pct, jnp.int32)
    full = jnp.asarray(full_canopy_day, jnp.int32)
    cap = jnp.where(jnp.asarray(determinant_crop, jnp.bool_),
                    jnp.asarray(flowering_day, jnp.int32) +
                    round_fortran(jnp.asarray(flowering_length, F64) / 2.0).astype(jnp.int32),
                    jnp.asarray(senescence_day, jnp.int32))

    def stressed(_):
        def days(reductions):
            red_g, red_c = reductions
            adjusted_maximum = (1.0 - red_c / 100.0) * maximum
            return days_to_reach_cover(
                0.98 * adjusted_maximum, initial, adjusted_maximum,
                growth * (1.0 - red_g / 100.0), emergence_day)

        state = (days((red_growth, red_cover)), red_growth, red_cover)

        def growth_step(state):
            _, red_g, red_c = state
            red_g -= 1
            return days((red_g, red_c)), red_g, red_c

        state = lax.while_loop(lambda s: (s[0] > cap) & (s[1] > 0),
                               growth_step, state)

        def cover_step(state):
            _, red_g, red_c = state
            red_c += 1
            return days((red_g, red_c)), red_g, red_c

        state = lax.while_loop(
            lambda s: (s[0] > cap) &
                      ((1.0 - s[2] / 100.0) * maximum > 0.10) &
                      (s[2] <= 50),
            cover_step, state)
        return state

    adjusted_day, red_growth, red_cover = lax.cond(
        (stress_class == 0) | ((red_growth == 0) & (red_cover == 0)),
        lambda _: (full, red_growth, red_cover), stressed, operand=None)
    return FertilityCanopyTiming(adjusted_day, red_growth, red_cover,
                                stress_class)


def soil_fertility_effect(stress_level_pct, shape_growth, shape_cover,
                          shape_water_productivity, shape_canopy_decline):
    """Translate ``CropStressParametersSoilFertility`` for scalar/batch inputs.

    Outputs retain Fortran units: rounded integer-like reductions are float64
    percent values; canopy_decline and the currently zero stomatal reduction
    are float64. Stress level and response-shape values broadcast together.
    """
    stress, growth, cover, wp, decline = jnp.broadcast_arrays(*[
        jnp.asarray(v, F64) for v in
        (stress_level_pct, shape_growth, shape_cover,
         shape_water_productivity, shape_canopy_decline)])
    relative_stress = stress / 100.0

    def effect(shape):
        coefficient = stress_coefficient(relative_stress, 0.0, 1.0, shape)
        return coefficient, round_fortran((1.0 - coefficient) * 100.0, bits=8)

    ks_growth, red_growth = effect(growth)
    ks_cover, red_cover = effect(cover)
    _, red_wp = effect(wp)
    ks_decline, decline_value = effect(decline)
    zeros = jnp.zeros_like(stress)
    return FertilityEffect(red_growth, red_cover, red_wp,
                           1.0 - ks_decline, zeros)
