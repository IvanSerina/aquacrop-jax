"""Bnormalized's reference calibration path, where WeedStress is zero."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .canopy import calculate_potential_et, fertility_canopy_no_water_stress
from .rounding import round_fortran

F64 = jnp.float64


class NormalizedBiomassTrajectory(NamedTuple):
    canopy: jax.Array
    transpiration_mm: jax.Array
    cumulative_kc: jax.Array
    water_productivity: jax.Array
    cumulative_biomass: jax.Array
    biomass: jax.Array


def normalized_biomass_trajectory(
        daily_gdd, calendar_days, thermal_days, initial_cover, maximum_cover,
        calendar_growth, thermal_growth, calendar_decline, thermal_decline,
        kc_top, kc_decline_pct, late_evaporation_pct, co2_ppm, minimum_gdd,
        *, sum_kc_top, full_canopy_sf_day, full_canopy_sf_gdd,
        days_per_gdd=1., reduction_growth_pct=0., reduction_cover_pct=0.,
        canopy_decline=0., water_productivity=15., fertility_stress_pct=0.,
        reduction_wp_pct=0., reproductive_crop=False, flowering_day=0,
        wp_yield_pct=100., yield_formation_days=0, switch_day=0,
        fertility_mode=True, reduction_stomatal_pct=0., gdd_mode=True,
        initial_cover_day=0, initial_cover_gdd=0):
    """Reference normalized biomass, with explicit static climate length.

    Milestones follow SeasonalSumOfKcPot. WP is the native g/m2 parameter,
    without the production routine's ton/ha conversion or CO2 WP adjustment.
    This is the weed-free path used by both fertility and salinity calibration.
    """
    daily_gdd = jnp.asarray(daily_gdd, F64)
    if daily_gdd.ndim != 1:
        raise ValueError("Reference climate must be a one-dimensional GDD series")
    if len(calendar_days) != 4 or len(thermal_days) != 4:
        raise ValueError("Reference calendars require four milestones")
    l0, full, senescence, maturity = [jnp.asarray(v, F64) for v in calendar_days]
    g0, gfull, gsenescence, gmaturity = [jnp.asarray(v, F64) for v in thermal_days]
    mode = jnp.asarray(gdd_mode, jnp.bool_)
    ini_day, ini_gdd = jnp.asarray(initial_cover_day, F64), jnp.asarray(initial_cover_gdd, F64)
    regrowth = ini_day != 0.
    tadj = jnp.where(ini_day == -9., full - l0, ini_day)
    gdd_tadj = jnp.where(ini_day == -9., gfull - g0, ini_gdd)
    start_gdd = jnp.where(regrowth, g0 + gdd_tadj, 0.)
    day_fraction = (senescence - full) / jnp.where(regrowth,
        tadj + l0 + (senescence - full), 1.)
    gdd_fraction = (gsenescence - gfull) / jnp.where(regrowth & mode,
        gdd_tadj + g0 + (gsenescence - gfull), 1.)
    sum_kc_target = (1. - jnp.asarray(fertility_stress_pct, F64) / 100.) * jnp.asarray(sum_kc_top, F64)
    max_sf = jnp.asarray(maximum_cover, F64) * (1. - jnp.asarray(reduction_cover_pct, F64) / 100.)
    cal_sf = (l0, full_canopy_sf_day, senescence, maturity)
    thermal_sf = (g0, full_canopy_sf_gdd, gsenescence, gmaturity)

    def step(state, forcing):
        total_gdd, from_day1, growth_on, withered, sum_kc, biomass = state
        day, daily = forcing
        total_gdd = jnp.where(mode, total_gdd + daily, total_gdd)
        from_day1 = jnp.where(mode, from_day1 + daily, from_day1)

        def first(_):
            starts = regrowth | jnp.where(mode, total_gdd > g0, day == l0 + 1.)
            return jnp.where(starts, jnp.asarray(initial_cover, F64), 0.), day, starts

        def developed(_):
            adjusted_day = jnp.minimum(day + tadj + l0, maturity)
            adjusted_day = jnp.where(adjusted_day > full,
                jnp.where(day <= senescence,
                    full + round_fortran(day_fraction * (day + tadj + l0 - full)),
                    day), adjusted_day)
            day_cc = jnp.where(regrowth, adjusted_day, day)
            plot_gdd = jnp.minimum(total_gdd, gmaturity)
            plot_gdd = jnp.where(plot_gdd > gfull,
                jnp.where(from_day1 <= gsenescence,
                    gfull + round_fortran(gdd_fraction * (from_day1 + gdd_tadj + g0 - gfull)),
                    from_day1), plot_gdd)
            plot_gdd = jnp.where(ini_gdd == 0., from_day1, plot_gdd)
            cc = fertility_canopy_no_water_stress(day_cc, plot_gdd,
                cal_sf, thermal_sf, initial_cover, maximum_cover,
                calendar_growth, thermal_growth, calendar_decline, thermal_decline,
                days_per_gdd, reduction_growth_pct, reduction_cover_pct,
                canopy_decline, gdd_mode=mode)
            return cc, day_cc, jnp.bool_(True)

        cc, day_cc, growth_on = lax.cond(growth_on, developed, first, None)
        withered = jnp.maximum(withered, cc)
        withered = jnp.where(day_cc >= full_canopy_sf_day, max_sf, withered)
        demand = calculate_potential_et(day_cc, l0, full, senescence, maturity,
            cc, 5., kc_top, kc_decline_pct, maximum_cover, withered,
            late_evaporation_pct, co2_ppm, daily, minimum_gdd)
        tpot = jnp.where(cc > .0001, demand.transpiration_mm, 0.)
        sum_kc += tpot / 5.

        switch = jnp.where((jnp.asarray(yield_formation_days) > 0) &
                           (jnp.asarray(switch_day) > 0),
            jnp.minimum((day - flowering_day) / jnp.where(jnp.asarray(switch_day) > 0,
                         switch_day, 1.), 1.), 1.)
        wp = jnp.asarray(water_productivity, F64) * jnp.where(
            jnp.asarray(reproductive_crop) & (jnp.asarray(wp_yield_pct) < 100.) &
            (day > flowering_day), 1. - (1. - wp_yield_pct / 100.) * switch, 1.)
        # The source deliberately spells epsilon(1.0), hence single precision.
        active = ((jnp.asarray(reduction_wp_pct) > 0.) & (sum_kc > 0.) &
                  (sum_kc_target > jnp.finfo(jnp.float32).eps))
        ratio = jnp.where(active, sum_kc / jnp.where(active, sum_kc_target, 1.), 1.)
        factor = jnp.where(sum_kc < sum_kc_target, jnp.exp(2. * jnp.log(ratio)), 1.)
        wp *= jnp.where(jnp.asarray(fertility_mode) & active,
                        1. - reduction_wp_pct / 100. * factor, 1.)
        stomatal = jnp.where(jnp.asarray(fertility_mode), 1., 1. - reduction_stomatal_pct / 100.)
        biomass += wp * stomatal * (tpot / 5.)
        return (total_gdd, from_day1, growth_on, withered, sum_kc, biomass), (
            cc, tpot, sum_kc, wp, biomass)

    initial = (jnp.where(mode, start_gdd, -9.), jnp.asarray(0., F64),
               regrowth, jnp.asarray(0., F64), jnp.asarray(0., F64), jnp.asarray(0., F64))
    final, (canopy, tpot, sum_kc, wp, biomass) = lax.scan(step, initial,
        (jnp.arange(1, daily_gdd.shape[0] + 1, dtype=F64), daily_gdd))
    return NormalizedBiomassTrajectory(canopy, tpot, sum_kc, wp, biomass, final[5])
