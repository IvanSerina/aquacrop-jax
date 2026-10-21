"""Unlimited-fertility biomass recurrence on actual daily crop climate."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .canopy import calculate_potential_et, potential_canopy_gdd
from .production import co2_water_productivity_factor
from .rounding import round_fortran
from .thermal_time import calendar_days_from_record


class PotentialBiomassParameters(NamedTuple):
    calendar_days: tuple
    thermal_days: tuple
    initial_cover: object
    maximum_cover: object
    thermal_growth: object
    thermal_decline: object
    kc_top: object
    kc_decline_pct: object
    late_evaporation_pct: object
    minimum_gdd: object
    water_productivity: object
    sink_strength_pct: object
    wet_bare_coefficient: object = 1.1
    senescence_exponent: object = 1.


class PotentialBiomassState(NamedTuple):
    biomass: jax.Array
    maximum_withered_cover: jax.Array


class PotentialBiomassDiagnostics(NamedTuple):
    canopy_cover: jax.Array
    thermal_calendar_day: jax.Array
    transpiration: jax.Array
    evaporation: jax.Array
    water_productivity: jax.Array
    increment: jax.Array
    valid: jax.Array


def potential_biomass_forage_gdd_step(state, cumulative_gdd, daily_gdd,
        eto_mm, co2_ppm, delayed_days, parameters, record_daily_gdd,
        first_offset=0, *, undefined_year=False):
    """DeterminePotentialBiomass for forage in GDD mode (no reproductive WP).

    Uses unlimited-fertility canopy/ET, independent of the actual stressed
    canopy and transpiration. Persistent withered cover updates even at ETo0;
    only biomass accumulation is guarded by ETo>0. Assimilate transfers and
    cutting resets belong to the following production/event operators.
    Invalid climate coverage preserves state and reports valid=False.
    """
    p = parameters
    g0, _, gsen, gmaturity = p.thermal_days
    cover = potential_canopy_gdd(cumulative_gdd, g0, gsen, gmaturity,
        p.initial_cover, p.maximum_cover, p.thermal_growth, p.thermal_decline)
    withered = jnp.maximum(jnp.asarray(state.maximum_withered_cover, jnp.float64), cover)
    converted = calendar_days_from_record(record_daily_gdd,
        round_fortran(jnp.asarray(cumulative_gdd, jnp.float64)), first_offset,
        undefined_year=undefined_year)
    dap = converted.value + jnp.asarray(delayed_days, jnp.int32)
    et = calculate_potential_et(dap, *p.calendar_days, cover, eto_mm,
        p.kc_top, p.kc_decline_pct, p.maximum_cover, withered,
        p.late_evaporation_pct, co2_ppm, daily_gdd, p.minimum_gdd,
        wet_bare_coefficient=p.wet_bare_coefficient, delayed_days=delayed_days,
        senescence_exponent=p.senescence_exponent)
    co2 = jnp.asarray(co2_ppm, jnp.float64)
    factor = co2_water_productivity_factor(co2, p.water_productivity, p.sink_strength_pct)
    factor = jnp.where(round_fortran(100. * co2) == round_fortran(jnp.float64(100. * 369.41)), 1., factor)
    wp = (jnp.asarray(p.water_productivity, jnp.float64) / 100.) * factor
    eto = jnp.asarray(eto_mm, jnp.float64)
    # Keep native WP*Tpot/ETo operation order, including the multiplication
    # before division; actual production uses a different expression.
    increment = jnp.where(eto > 0., (wp * et.transpiration_mm) / jnp.where(eto > 0., eto, 1.), 0.)
    biomass = jnp.asarray(state.biomass, jnp.float64)
    valid = converted.valid
    next_state = PotentialBiomassState(jnp.where(valid, biomass + increment, biomass),
        jnp.where(valid, withered, state.maximum_withered_cover))
    return next_state, PotentialBiomassDiagnostics(cover, dap,
        et.transpiration_mm, et.evaporation_mm, wp, increment, valid)
