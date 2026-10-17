"""Pure JAX translations of AquaCrop numerical operators."""

from __future__ import annotations

import jax
import jax.numpy as jnp

F64 = jnp.float64


@jax.jit
def degrees_day(tbase, tupper, tmin, tmax, method):
    tbase, tupper, tmin, tmax = [jnp.asarray(x, F64) for x in
                                 (tbase, tupper, tmin, tmax)]
    method = jnp.asarray(method, jnp.int32)
    avg1 = jnp.clip((tmax + tmin) / 2.0, tbase, tupper)
    avg2 = (jnp.clip(tmax, tbase, tupper) +
            jnp.clip(tmin, tbase, tupper)) / 2.0
    avg3 = jnp.maximum((jnp.clip(tmax, tbase, tupper) +
                        jnp.minimum(tmin, tupper)) / 2.0, tbase)
    return jnp.where(method == 1, avg1, jnp.where(method == 2, avg2, avg3)) - tbase


@jax.jit
def curve_number_runoff(rain, curve_number, initial_abstraction_pct=5.0):
    rain = jnp.asarray(rain, F64)
    cn = jnp.asarray(curve_number, F64)
    abstraction = jnp.asarray(initial_abstraction_pct, F64) / 100.0
    retention = 254.0 * (100.0 / cn - 1.0)
    term = rain - abstraction * retention
    runoff = term * term / (rain + (1.0 - abstraction) * retention)
    return jnp.where(term <= jnp.finfo(F64).eps, jnp.zeros_like(runoff), runoff)


@jax.jit
def compartment_water_mm(theta, thickness_m, gravel_fraction):
    return (1000.0 * jnp.asarray(theta, F64) *
            jnp.asarray(thickness_m, F64) *
            (1.0 - jnp.asarray(gravel_fraction, F64)))


@jax.jit
def biomass_increment(water_productivity, actual_transpiration, eto):
    wp, tr, eto = [jnp.asarray(x, F64) for x in
                   (water_productivity, actual_transpiration, eto)]
    return jnp.where(eto > 0.0, wp * tr / eto, jnp.zeros_like(eto))
