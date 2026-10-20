"""Canopy-cover state recurrences and evapotranspiration operators."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .rounding import round_fortran
from .stress import stress_coefficient
F64 = jnp.float64


class PotentialET(NamedTuple):
    transpiration_mm: jax.Array
    evaporation_mm: jax.Array
    adjusted_canopy: jax.Array
    crop_coefficient: jax.Array
    cold_stress: jax.Array


class ProtectedSeedlingState(NamedTuple):
    """Persistent ``CCiPrev`` and ``Simulation.ProtectedSeedling`` state."""
    cover: jax.Array
    protected: jax.Array


class EarlyVegetativeCanopyState(NamedTuple):
    """Fortran CCiPrev, CCoAdjusted, CCxAdjusted and ProtectedSeedling."""
    cover: jax.Array
    initial_adjusted: jax.Array
    maximum_adjusted: jax.Array
    protected: jax.Array


class WaterSenescenceState(NamedTuple):
    """Persistent state in the water-senescence part of DetermineCCiGDD."""
    time_gdd: jax.Array
    top_cover: jax.Array
    withered_cover: jax.Array
    eto_stress_sum_mm: jax.Array
    evaporation_limited: jax.Array
    no_more_crop: jax.Array


def senescence_threshold_for_eto(eto_mm, default_threshold, time_gdd,
                                 fao_correction=True, adjustment_factor=1.0,
                                 beta_pct=10.0, with_beta=True):
    """AdjustpSenescenceToETo; beta and clipping belong only to FAOCorrection."""
    eto, threshold, time, factor, beta = [jnp.asarray(v, F64) for v in
        (eto_mm, default_threshold, time_gdd, adjustment_factor, beta_pct)]

    def adjusted(_):
        value = threshold + factor * 0.04 * (5.0 - eto) * jnp.log10(
            10.0 - 9.0 * threshold)
        value = jnp.where((time > 0.0001) & with_beta,
                          value * (1.0 - beta / 100.0), value)
        return jnp.where(value >= 1.0, 0.98, jnp.maximum(value, 0.0))

    return lax.cond(jnp.asarray(fao_correction, jnp.bool_), adjusted,
                    lambda _: threshold, operand=None)


def senescence_gdd_decline_coefficient(relative_depletion, p_senescence,
                                       shape_factor, decline_coefficient,
                                       maximum_cover, stress_maximum_cover):
    """DetermineGDDCDCadjustedWaterStress with its native KsSen guard.

    p_senescence must be the ETo-adjusted threshold *without* beta. Native
    roundoff can produce a zero coefficient just above that threshold;
    the caller must reject a singular log-time recurrence, not move pSen.
    """
    relative, threshold, shape, decline, maximum, stress_maximum = [
        jnp.asarray(v, F64) for v in (relative_depletion, p_senescence,
            shape_factor, decline_coefficient, maximum_cover, stress_maximum_cover)]
    full = decline * ((stress_maximum + 2.29) / (maximum + 2.29))

    def dry(_):
        def partial(_):
            ks = stress_coefficient(relative, threshold, 0.999, shape)
            return lax.cond(ks > 1e-6,
                lambda _: (full * (1.0 - jnp.exp(8.0 * jnp.log(ks))),
                           100.0 * (1.0 - ks)),
                lambda _: (jnp.asarray(.0001, F64), jnp.asarray(0., F64)), None)

        return lax.cond(relative >= .999,
                        lambda _: (full, jnp.asarray(100., F64)), partial, None)

    return lax.cond(relative <= threshold,
                    lambda _: (jnp.asarray(.0001, F64), jnp.asarray(0., F64)), dry, None)


def leaf_thresholds_for_eto(eto_mm, default_upper, default_lower,
                            fao_correction=True, adjustment_factor=1.0):
    """``AdjustpLeafToETo`` before selecting root-zone canopy stress."""
    eto, upper, lower, factor = [jnp.asarray(v, F64) for v in
        (eto_mm, default_upper, default_lower, adjustment_factor)]
    def adjusted(p):
        return jnp.clip(p + factor * 0.04 * (5.0 - eto) *
                        jnp.log10(10.0 - 9.0 * p), 0.0, 1.0)
    return (jnp.where(fao_correction, adjusted(upper), upper),
            jnp.where(fao_correction, adjusted(lower), lower))


def leaf_expansion_gdd_coefficient(growth_coefficient, actual_water_mm,
                                   field_capacity_mm, wilting_point_mm,
                                   p_leaf_upper, p_leaf_lower, shape_factor):
    """``DetermineGDDCGCadjusted`` using the selected root/topsoil zone."""
    growth, actual, fc, wp, upper, lower, shape = [jnp.asarray(v, F64) for v in
        (growth_coefficient, actual_water_mm, field_capacity_mm,
         wilting_point_mm, p_leaf_upper, p_leaf_lower, shape_factor)]
    depletion = jnp.where(fc != wp, (fc - actual) / (fc - wp), 0.0)
    ks = stress_coefficient(depletion, upper, lower, shape)
    ks = jnp.where(actual >= fc, 1.0,
                   jnp.where(actual <= wp, 0.0,
                             jnp.where(depletion <= upper, 1.0,
                                       jnp.where(depletion >= lower, 0.0, ks))))
    return growth * ks, 100.0 * (1.0 - ks)


def required_gdd(previous_cover, initial_cover, maximum_cover,
                 adjusted_growth, cumulative_gdd, daily_gdd):
    """``DetermineCCiGDD::RequiredGDD`` with its source operation order."""
    previous, initial, maximum, growth, total, daily = [jnp.asarray(v, F64)
        for v in (previous_cover, initial_cover, maximum_cover,
                  adjusted_growth, cumulative_gdd, daily_gdd)]
    elapsed = total - daily
    early = jnp.log(previous / initial) / elapsed
    late = jnp.log((0.25 * maximum * maximum / initial) /
                   (maximum - previous)) / elapsed
    cgcx = jnp.where(previous <= maximum / 2.0, early, late)
    return elapsed * cgcx / growth


def early_vegetative_gdd_step(
        state, cumulative_gdd, daily_gdd, emergence_gdd, full_canopy_sf_gdd,
        final_growth_gdd, senescence_gdd, maturity_gdd, initial_cover,
        maximum_cover, growth_coefficient, decline_coefficient,
        reduction_growth_pct, reduction_cover_pct, effective_water_mm,
        effective_fc_mm, effective_wp_mm, p_leaf_upper, p_leaf_lower,
        leaf_shape_factor):
    """Early ``DetermineCCiGDD`` through the full-canopy stress milestone.

    The caller supplies the wetter selected root/topsoil zone *after*
    infiltration, capillary rise and salt transport but *before* ET. A false
    validity flag forbids use during fertility canopy decline or senescence.
    """
    gdd, daily, initial, maximum = [jnp.asarray(v, F64) for v in
        (cumulative_gdd, daily_gdd, initial_cover, maximum_cover)]
    full_sf = jnp.asarray(full_canopy_sf_gdd, F64)
    final = jnp.asarray(final_growth_gdd, F64)
    senescence = jnp.asarray(senescence_gdd, F64)
    protected = jnp.asarray(state.protected, jnp.bool_)
    previous = jnp.asarray(state.cover, F64)
    initial_adjusted = jnp.asarray(state.initial_adjusted, F64)
    before = gdd <= jnp.asarray(emergence_gdd, F64)
    after = round_fortran(gdd) > jnp.asarray(maturity_gdd, F64)
    active = (gdd < final) & (gdd < senescence) & (gdd <= full_sf)
    valid = before | after | active
    growth_sf = jnp.asarray(growth_coefficient, F64) * (
        1.0 - jnp.asarray(reduction_growth_pct, F64) / 100.0)
    maximum_sf = maximum * (1.0 - jnp.asarray(reduction_cover_pct, F64) / 100.0)

    def growing(_):
        first = (previous <= initial_adjusted) | (gdd <= daily) | (
            protected & (previous <= 1.25 * initial))

        def first_day(_):
            potential = potential_canopy_gdd(
                gdd, emergence_gdd, senescence_gdd, maturity_gdd,
                initial, maximum, growth_coefficient, decline_coefficient,
                reduction_growth_pct, reduction_cover_pct)
            canopy = jnp.where(protected, potential,
                               initial_adjusted * jnp.exp(growth_sf * daily))
            next_protected = protected & (canopy <= 1.25 * initial)
            return (canopy, initial_adjusted, canopy, next_protected,
                    jnp.asarray(-9.0, F64))

        def later_day(_):
            def below_maximum(_):
                adjusted_growth, stress = leaf_expansion_gdd_coefficient(
                    growth_sf, effective_water_mm, effective_fc_mm,
                    effective_wp_mm, p_leaf_upper, p_leaf_lower,
                    leaf_shape_factor)

                def still_growing(_):
                    elapsed = required_gdd(previous, initial_adjusted,
                                           maximum_sf, adjusted_growth,
                                           gdd, daily)
                    extended = elapsed + (final - gdd) + daily
                    adjusted_maximum = jnp.where(elapsed > 0.0,
                        canopy_at_time(extended, initial_adjusted,
                                       adjusted_growth, maximum_sf), -9.0)
                    near_maximum = jnp.abs(previous - .97999 * maximum_sf) < .001
                    potential = potential_canopy_gdd(
                        gdd, emergence_gdd, senescence_gdd, maturity_gdd,
                        initial, maximum, growth_coefficient,
                        decline_coefficient, reduction_growth_pct,
                        reduction_cover_pct)
                    candidate = canopy_at_time(
                        elapsed + daily, initial_adjusted,
                        adjusted_growth, adjusted_maximum)
                    canopy = jnp.where(adjusted_maximum < 0.0, previous,
                                       jnp.where(near_maximum, potential,
                                                 jnp.where(elapsed < 0.0,
                                                           previous, candidate)))
                    return canopy, initial_adjusted, stress

                def no_growth(_):
                    adjusted_initial = jnp.where(previous > initial_adjusted,
                                                 initial, previous)
                    return previous, adjusted_initial, stress

                return lax.cond(adjusted_growth > 1e-6, still_growing,
                                no_growth, operand=None)

            def at_maximum(_):
                potential = potential_canopy_gdd(
                    gdd, emergence_gdd, senescence_gdd, maturity_gdd,
                    initial, maximum, growth_coefficient, decline_coefficient,
                    reduction_growth_pct, reduction_cover_pct)
                return potential, initial, jnp.asarray(-33.0, F64)

            canopy, adjusted_initial, stress = lax.cond(
                previous < .97999 * maximum_sf, below_maximum,
                at_maximum, operand=None)
            capped = canopy > maximum_sf
            canopy = jnp.minimum(canopy, maximum_sf)
            stress = jnp.where(capped, -33.0, stress)
            return canopy, adjusted_initial, canopy, protected, stress

        cover, cco, ccx, seedling, stress = lax.cond(
            first, first_day, later_day, operand=None)
        return EarlyVegetativeCanopyState(cover, cco, ccx, seedling), stress

    def inactive(_):
        cover = jnp.where(before | after, 0.0, previous)
        return EarlyVegetativeCanopyState(
            cover, initial_adjusted, state.maximum_adjusted, protected), jnp.asarray(-9.0, F64)

    next_state, stress = lax.cond(active & ~(before | after), growing,
                                  inactive, operand=None)
    return next_state, stress, valid


def early_water_senescence_gdd_step(
        state, canopy, previous_cover, cumulative_gdd, daily_gdd,
        emergence_gdd, senescence_gdd, final_growth_gdd,
        initial_cover, crop_initial_cover, maximum_cover, stress_maximum_cover,
        decline_coefficient, effective_water_mm, effective_fc_mm,
        effective_wp_mm, eto_mm, p_senescence, senescence_shape_factor,
        delay_eto_mm, *, canopy_valid, fao_correction=True,
        adjustment_factor=1.0, beta_pct=10.0):
    """Apply DetermineCCiGDD steps 4, 5 and 7 before natural senescence.

    ``canopy`` is today's vegetative result; ``previous_cover`` is yesterday's
    CCiPrev. The effective water zone is selected after pre-ET water/salt
    transport. CCoTotal and Crop.CCo are distinct inputs (e.g. with weeds).
    A false validity flag forbids consuming the returned state, including
    the native near-threshold singularity where KsSen rounds to one.
    Late-season decline and rewatering are deliberately outside this operator.
    """
    (previous, gdd, daily, emergence, senescence, final, initial, crop_initial,
     maximum, stress_maximum, decline, actual, fc, wp, eto, psen, shape,
     delay) = [jnp.asarray(v, F64) for v in (
        previous_cover, cumulative_gdd, daily_gdd, emergence_gdd,
        senescence_gdd, final_growth_gdd, initial_cover, crop_initial_cover,
        maximum_cover, stress_maximum_cover, decline_coefficient,
        effective_water_mm, effective_fc_mm, effective_wp_mm, eto_mm,
        p_senescence, senescence_shape_factor, delay_eto_mm)]
    canopy = EarlyVegetativeCanopyState(
        *[jnp.asarray(v, F64) for v in canopy[:3]], jnp.asarray(canopy.protected, jnp.bool_))
    state = WaterSenescenceState(
        *[jnp.asarray(v, F64) for v in state[:4]],
        jnp.asarray(state.evaporation_limited, jnp.bool_),
        jnp.asarray(state.no_more_crop, jnp.bool_))
    supported = (jnp.asarray(canopy_valid, jnp.bool_) & (gdd > emergence)
                 & (gdd < senescence) & (fc > wp))

    def invalid(_):
        return canopy, state, jnp.asarray(-9., F64), jnp.asarray(False)

    def active(_):
        threshold = senescence_threshold_for_eto(
            eto, psen, state.time_gdd, fao_correction, adjustment_factor, beta_pct)
        onset = (actual < fc - threshold * (fc - wp)) & ~canopy.protected

        def dry(_):
            top = jnp.where(jnp.abs(state.time_gdd) < jnp.finfo(F64).eps,
                            canopy.cover, state.top_cover)
            time = state.time_gdd + daily
            coefficient_threshold = senescence_threshold_for_eto(
                eto, psen, time, fao_correction, adjustment_factor, beta_pct,
                with_beta=False)
            cdc, stress = senescence_gdd_decline_coefficient(
                (fc - actual) / (fc - wp), coefficient_threshold, shape,
                decline, maximum, stress_maximum)

            def dormant_cover(_):
                return jnp.where(.05 > crop_initial,
                    crop_initial + (1.0 - state.eto_stress_sum_mm / delay) *
                    (.05 - crop_initial), crop_initial)

            def exhausted(_):
                exhausted = ((state.eto_stress_sum_mm > delay)
                             | (jnp.abs(delay) < jnp.finfo(F64).eps) | (gdd >= final))
                return lax.cond(exhausted, lambda _: jnp.asarray(0., F64),
                                dormant_cover, None)

            def normal_top(_):
                guarded = ((time * cdc * 3.33 / (top + 2.29) > 100.0)
                           | (previous >= 1.05 * top))

                def logarithmic(_):
                    argument = 1.0 + (1.0 - previous / top) / .05
                    regular = (cdc > 0.0) & (argument > 0.0)

                    def calculate(_):
                        elapsed = jnp.log(argument) / (cdc * 3.33 / (top + 2.29))
                        value = top * (1.0 - .05 * (
                            jnp.exp((elapsed + daily) * cdc * 3.33 / (top + 2.29)) - 1.0))
                        return value, jnp.isfinite(value)

                    return lax.cond(regular, calculate,
                                    lambda _: (jnp.asarray(0., F64), jnp.asarray(False)), None)

                candidate, regular = lax.cond(guarded,
                    lambda _: (exhausted(None), jnp.asarray(True)), logarithmic, None)
                candidate = jnp.maximum(candidate, 0.0)
                floor = ((delay > 0.0) & (state.eto_stress_sum_mm <= delay)
                         & (gdd < final) & ((candidate < crop_initial) | (candidate < .05)))
                candidate = lax.cond(floor, dormant_cover, lambda _: candidate, None)
                return candidate, regular

            candidate, regular = lax.cond(top < .001,
                lambda _: (exhausted(None), jnp.asarray(True)), normal_top, None)
            candidate = jnp.minimum(candidate, stress_maximum)
            cover = jnp.minimum(candidate, previous)
            next_canopy = EarlyVegetativeCanopyState(
                cover, jnp.minimum(cover, initial), cover, canopy.protected)
            # Native accumulation uses CCiSen before the CCiPrev no-jump cap.
            low = ((round_fortran(10000.0 * candidate) <= 500.0)
                   | (round_fortran(10000.0 * candidate) <=
                      round_fortran(10000.0 * crop_initial)))
            next_state = WaterSenescenceState(
                time, top, state.withered_cover,
                state.eto_stress_sum_mm + jnp.where(low, eto, 0.0),
                jnp.asarray(True), state.no_more_crop)
            return next_canopy, next_state, stress, regular

        def wet(_):
            return canopy, WaterSenescenceState(
                jnp.asarray(0., F64), state.top_cover, state.withered_cover,
                jnp.asarray(0., F64), state.evaporation_limited,
                state.no_more_crop), jnp.asarray(0., F64), jnp.asarray(True)

        next_canopy, next_state, stress, regular = lax.cond(onset, dry, wet, None)
        next_state = next_state._replace(
            withered_cover=jnp.maximum(next_state.withered_cover, next_canopy.cover),
            no_more_crop=(next_state.no_more_crop |
                          (round_fortran(1000.0 * next_canopy.cover) <= 0.0)))
        return lax.cond(regular, lambda _: (next_canopy, next_state, stress, regular),
                        invalid, None)

    return lax.cond(supported, active, invalid, None)


def protected_seedling_gdd_step(state, cumulative_gdd, daily_gdd,
                               emergence_gdd, senescence_gdd, maturity_gdd,
                               initial_cover, maximum_cover, growth_coefficient,
                               decline_coefficient, reduction_growth_pct,
                               reduction_cover_pct):
    """The protected-seedling branch of ``DetermineCCiGDD``.

    The returned validity flag becomes false on the *next* day after cover
    exceeds 1.25 CCo. The caller must switch to the water-stressed canopy
    recurrence then; this operator deliberately has no fallback curve.
    """
    gdd = jnp.asarray(cumulative_gdd, F64)
    previous = jnp.asarray(state.cover, F64)
    protected = jnp.asarray(state.protected, jnp.bool_)
    before_emergence = gdd <= jnp.asarray(emergence_gdd, F64)
    after_harvest = round_fortran(gdd) > jnp.asarray(maturity_gdd, F64)
    growing = gdd < jnp.asarray(senescence_gdd, F64)
    # DetermineCCiGDD enters its first-growth branch while the seedling is
    # protected and CCiPrev <= 1.25 CCo, regardless of leaf water stress.
    initial = jnp.asarray(initial_cover, F64)
    first_growth = (previous <= initial) | (gdd <= jnp.asarray(daily_gdd, F64))
    phase_valid = before_emergence | after_harvest | (
        growing & protected & ((previous <= 1.25 * initial) | first_growth))
    potential = potential_canopy_gdd(
        gdd, emergence_gdd, senescence_gdd, maturity_gdd,
        initial, maximum_cover, growth_coefficient, decline_coefficient,
        reduction_growth_pct, reduction_cover_pct)
    cover = jnp.where(before_emergence | after_harvest, 0.0,
                      jnp.where(phase_valid, potential, previous))
    still_protected = protected & (cover <= 1.25 * initial)
    return ProtectedSeedlingState(cover, still_protected), phase_valid


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
    active = jnp.logical_and(gdd > 0.0, round_fortran(gdd) <= maturity)
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
    # KsTemperature bypasses undefined (-9 after roundc) or coincident limits.
    # ac_global::eps is initialized from a default-real literal (10E-08).
    undefined = jnp.logical_or(round_fortran(lower) == -9., round_fortran(upper) == -9.)
    bypass = jnp.logical_or(undefined,
        jnp.abs(lower - upper) <= jnp.asarray(jnp.float32(1e-7), F64))
    return jnp.where(bypass, 1.0,
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
    adjusted = jnp.where(adjusted < jnp.finfo(F64).eps, 0.0, adjusted)
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
    cold = jnp.where(jnp.logical_or(adjusted <= 1e-6, round_fortran(gdd) < 0.0),
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
    minimum_evaporation = jnp.where(
        minimum_evaporation < jnp.finfo(F64).eps, 0.0, minimum_evaporation)
    late_evaporation = jnp.clip(
        late_evaporation, minimum_evaporation, maximum_evaporation)
    evaporation = jnp.where(jnp.logical_and(day > senescence, maximum > jnp.finfo(F64).eps),
                           late_evaporation, evaporation)
    evaporation = jnp.where(jnp.asarray(evaporation_limit, jnp.bool_),
                            jnp.minimum(evaporation, maximum_evaporation), evaporation)
    dying = jnp.logical_and(canopy < withered,
                            jnp.logical_and(withered > 0.01, canopy > 0.001))
    transpiration *= jnp.where(
        dying, jnp.exp(jnp.asarray(senescence_exponent, F64) *
                       jnp.log(canopy / withered)), 1.0)
    outside = jnp.logical_or(
        jnp.logical_and(day < emergence, round_fortran(100.0 * canopy) == 0.0),
        day > harvest)
    transpiration = jnp.where(outside, 0.0, transpiration)
    evaporation = jnp.where(outside, bare * eto, evaporation)
    return PotentialET(transpiration, evaporation, adjusted, kc, cold)
