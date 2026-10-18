"""Crop response to soil fertility stress, independent of event state."""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .rounding import round_fortran
from .transpiration import stress_coefficient

F64 = jnp.float64


class FertilityEffect(NamedTuple):
    reduction_growth_pct: jax.Array
    reduction_cover_pct: jax.Array
    reduction_water_productivity_pct: jax.Array
    canopy_decline: jax.Array
    reduction_stomatal_pct: jax.Array


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
