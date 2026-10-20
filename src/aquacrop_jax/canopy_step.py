"""Compose the verified GDD canopy recurrence on the pre-ET water state."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .canopy import (EarlyVegetativeCanopyState, LateCanopyParameters,
    WaterSenescenceState, early_vegetative_gdd_step,
    early_water_senescence_gdd_step, fertility_canopy_ceiling_gdd,
    leaf_thresholds_for_eto, nonexpanding_canopy_gdd_step,
    senescence_threshold_for_eto)
from .rounding import round_fortran

F64 = jnp.float64


class CanopyDayState(NamedTuple):
    canopy: EarlyVegetativeCanopyState
    senescence: WaterSenescenceState


class CanopyDayParameters(NamedTuple):
    # Milestones: emergence, fertility full canopy, senescence, maturity.
    calendar_days: tuple
    thermal_days: tuple
    final_growth_gdd: jax.Array
    initial_cover: jax.Array
    crop_initial_cover: jax.Array
    maximum_cover: jax.Array
    calendar_growth: jax.Array
    thermal_growth: jax.Array
    calendar_decline: jax.Array
    thermal_decline: jax.Array
    leaf_upper: jax.Array
    leaf_lower: jax.Array
    leaf_shape: jax.Array
    p_senescence: jax.Array
    senescence_shape: jax.Array
    delay_eto_mm: jax.Array
    fao_correction: jax.Array
    adjustment_factor: jax.Array
    beta_pct: jax.Array


class CanopyDayDiagnostics(NamedTuple):
    leaf_stress_pct: jax.Array
    senescence_stress_pct: jax.Array
    senescence_threshold: jax.Array
    fertility_ceiling: jax.Array
    valid: jax.Array


def active_canopy_gdd_step(state, cumulative_gdd, daily_gdd, calendar_day,
        eto_mm, effective_water_mm, effective_fc_mm, effective_wp_mm,
        parameters, effect):
    """Native DetermineCCiGDD growth -> water senescence -> final guards.

    The caller supplies the wetter root/topsoil zone after water/salt inputs,
    the integrated fertility/salinity effect, and current SF milestones.
    ``calendar_day`` is VirtualTimeCC + DelayedDays + 1; final growth is the
    native determinacy/regrowth-adjusted milestone, computed by the caller.
    This active-season operator rejects emergence/out-of-season states and
    known singularities with valid=False and preserves the previous state.
    Germination delays, cutting resets, root growth and ET are separate steps.
    """
    p = parameters
    gdd = jnp.asarray(cumulative_gdd, F64)
    emergence, full_sf, senescence, maturity = p.thermal_days
    supported = ((gdd > emergence) & (round_fortran(gdd) <= maturity)
                 & (p.final_growth_gdd <= senescence))

    def invalid(_):
        unset = jnp.asarray(-9., F64)
        return state, CanopyDayDiagnostics(unset, unset, unset, unset, jnp.bool_(False))

    def advance(_):
        cap = fertility_canopy_ceiling_gdd(calendar_day, gdd,
            p.calendar_days, p.thermal_days, p.initial_cover, p.maximum_cover,
            p.calendar_growth, p.thermal_growth, p.calendar_decline, p.thermal_decline,
            effect.reduction_growth_pct, effect.reduction_cover_pct, effect.canopy_decline)
        upper, lower = leaf_thresholds_for_eto(eto_mm, p.leaf_upper, p.leaf_lower,
            p.fao_correction, p.adjustment_factor)
        maximum_sf = p.maximum_cover * (1. - effect.reduction_cover_pct / 100.)
        ratio = jnp.where(full_sf < senescence,
            (p.calendar_days[2] - p.calendar_days[1]) /
            jnp.where(full_sf < senescence, senescence - full_sf, 1.), 1.)
        developing, leaf, valid = lax.cond(gdd < p.final_growth_gdd,
            lambda _: early_vegetative_gdd_step(state.canopy, gdd, daily_gdd,
                emergence, full_sf, p.final_growth_gdd, senescence, maturity,
                p.initial_cover, p.maximum_cover, p.thermal_growth, p.thermal_decline,
                effect.reduction_growth_pct, effect.reduction_cover_pct,
                effective_water_mm, effective_fc_mm, effective_wp_mm,
                upper, lower, p.leaf_shape, fertility_decline_cap=cap),
            lambda _: nonexpanding_canopy_gdd_step(state.canopy, gdd,
                p.final_growth_gdd, full_sf, senescence, maturity, p.maximum_cover,
                maximum_sf, p.thermal_decline, cap, effect.canopy_decline, ratio), None)
        threshold = senescence_threshold_for_eto(eto_mm, p.p_senescence,
            state.senescence.time_gdd, p.fao_correction, p.adjustment_factor, p.beta_pct)
        late = LateCanopyParameters(maturity, p.thermal_growth,
            effect.reduction_growth_pct, effect.reduction_cover_pct)
        canopy, senescence_state, stress, valid = early_water_senescence_gdd_step(
            state.senescence, developing, state.canopy.cover, gdd, daily_gdd,
            emergence, senescence, p.final_growth_gdd, p.initial_cover,
            p.crop_initial_cover, p.maximum_cover, maximum_sf, p.thermal_decline,
            effective_water_mm, effective_fc_mm, effective_wp_mm, eto_mm,
            p.p_senescence, p.senescence_shape, p.delay_eto_mm, canopy_valid=valid,
            fao_correction=p.fao_correction, adjustment_factor=p.adjustment_factor,
            beta_pct=p.beta_pct, fertility_decline_cap=cap, late_canopy=late)
        next_state = CanopyDayState(canopy, senescence_state)
        next_state = lax.cond(valid, lambda _: next_state, lambda _: state, None)
        return next_state, CanopyDayDiagnostics(leaf, stress, threshold, cap, valid)

    return lax.cond(supported, advance, invalid, None)
