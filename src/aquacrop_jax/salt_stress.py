"""Explicit crop salt-stress effects from tempprocessing.f90.

These operators expose the calibrated canopy/stomatal reductions. Coupling to
actual canopy feedback and TimeToMaxCanopySF remains a separate runner step.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .salinity import F64, round_fortran, salinity_stress


class SaltCropEffect(NamedTuple):
    growth_reduction_pct: jax.Array
    cover_reduction_pct: jax.Array
    productivity_reduction_pct: jax.Array
    canopy_decline_pct_day: jax.Array
    stomatal_reduction_pct: jax.Array


def salt_canopy_effect(reduction_pct, distortion_pct, initial_cover, maximum_cover,
                       calendar_growth, gdd_growth, full_day, flowering_day,
                       flowering_days, harvest_day, full_gdd, flowering_gdd,
                       flowering_length_gdd, harvest_gdd, determinate=False,
                       gdd_mode=True):
    """tempprocessing.f90 CropStressParametersSoilSalinity, including GDD ramp."""
    red, distortion, initial, maximum, cgc, gddcgc, full, flower, flower_days, harvest, gfull, gflower, gflower_days, gharvest = [
        jnp.asarray(x, F64) for x in (reduction_pct, distortion_pct, initial_cover,
            maximum_cover, calendar_growth, gdd_growth, full_day, flowering_day,
            flowering_days, harvest_day, full_gdd, flowering_gdd,
            flowering_length_gdd, harvest_gdd)]
    gdd = jnp.asarray(gdd_mode, jnp.bool_)
    base_case = (initial > .98 * maximum) | (initial >= maximum) | (red == 0.)
    log_reference = jnp.log(12.5 * maximum / initial)
    reference_calendar = jnp.where(base_case | gdd, full, log_reference / cgc)
    reference_gdd = log_reference / gddcgc
    ref_time = jnp.where(gdd, reference_gdd, reference_calendar)
    growth = jnp.where(gdd, gddcgc, cgc)
    adjusted_maximum = .90 * maximum * (1. - red / 100.)
    active = ~base_case & (ref_time > jnp.finfo(F64).eps) & (.02 * adjusted_maximum >= .0001)
    time = jnp.where(gdd, gfull, full)
    end = jnp.where(gdd, gharvest, harvest)
    flower_end = jnp.where(gdd, gflower + round_fortran(gflower_days / 2.),
                           flower + round_fortran(flower_days / 2.))
    maximum_time = time + (end - time) / 2.
    maximum_time = jnp.where(jnp.asarray(determinate) & (maximum_time > flower_end),
                             flower_end, maximum_time)
    log_adjusted = jnp.log(jnp.maximum(12.5 * adjusted_maximum / initial, 1e-30))
    maximum_growth = log_adjusted / ref_time
    minimum_growth = jnp.where(maximum_time > ref_time,
                               log_adjusted / maximum_time, maximum_growth)
    ramp = jnp.where(red < 10., jnp.where(gdd, jnp.exp(red - 10.), 1.5 ** (red - 10.)), 1.)
    adjusted_growth = maximum_growth - (maximum_growth - minimum_growth) * ramp * distortion / 100.
    reduction = jnp.where(active, round_fortran(100. * (growth - adjusted_growth) / growth), 0.)
    # The reference's decline block uses calendar L12 variables even in GDD mode.
    calendar_maximum_time = jnp.where(active & ~gdd, maximum_time, full)
    adjusted_cover = .98 * maximum * (1. - red / 100.)
    decline_start = calendar_maximum_time - (calendar_maximum_time - reference_calendar) * distortion / 100.
    final_cover = jnp.maximum(initial, adjusted_cover -
        jnp.where(red < 10., 1.5 ** (red - 10.), 1.) * .5 * distortion / 100. * (adjusted_cover - initial))
    decline = 100. * (adjusted_cover - final_cover) / jnp.where(harvest != decline_start, harvest - decline_start, 1.)
    decline = jnp.minimum(decline, 1.)
    decline = jnp.where(decline <= jnp.finfo(F64).eps, .001, decline)
    decline = jnp.where((harvest > decline_start) & (distortion > 0.), decline, .001)
    decline = jnp.where(red == 0., 0., decline)
    return SaltCropEffect(reduction, red, jnp.asarray(0., F64), decline, red)


def run_salt_stress_average(rootzone_ece, growing, minimum_ece, maximum_ece,
                             coefficients, initial_days=0., initial_stress_pct=0.):
    """Explicit running mean and calibrated CCx reductions used by Effect...Stress.

    The caller supplies the germinated growing-day mask and resets this state at
    the crop boundary. Coefficients come from the reference biomass calibration.
    """
    ece = jnp.asarray(rootzone_ece, F64)
    growing = jnp.asarray(growing, jnp.bool_)
    if ece.shape != growing.shape:
        raise ValueError('Salt ECe and growing-day mask must have equal shape')

    def step(carry, inputs):
        days, previous = carry
        ec, active = inputs
        current = 100. * (1. - salinity_stress(ec, minimum_ece, maximum_ece))
        mean = (days * previous + current) / (days + 1.)
        mean = jnp.where(active, mean, previous)
        b0, b1, b2 = coefficients
        reduction = round_fortran(b0 + b1 * mean + b2 * mean * mean)
        reduction = jnp.where((reduction < 0.) | (mean <= .1), 0.,
                              jnp.where((reduction > 100.) | (mean >= 99.9), 100., reduction))
        return (days + active.astype(F64), mean), (mean, jnp.where(active, reduction, 0.))

    _, output = lax.scan(step, (jnp.asarray(initial_days, F64),
                               jnp.asarray(initial_stress_pct, F64)), (ece, growing))
    return output
