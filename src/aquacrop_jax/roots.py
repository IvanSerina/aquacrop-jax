"""JAX root-depth and root-zone water operators."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

F64 = jnp.float64


class RootZoneWater(NamedTuple):
    actual_mm: jax.Array
    field_capacity_mm: jax.Array
    wilting_point_mm: jax.Array
    saturation_mm: jax.Array
    depletion_mm: jax.Array
    total_available_mm: jax.Array
    relative_depletion: jax.Array
    fractions: jax.Array


@jax.jit
def time_root_function(time, shape_factor, maximum_time, initial_time):
    time, shape, maximum, initial = [jnp.asarray(v, F64) for v in
                                     (time, shape_factor, maximum_time,
                                      initial_time)]
    return jnp.exp((10.0 / shape) * jnp.log(
        (time - initial) / (maximum - initial)))


@jax.jit
def actual_rooting_depth_gdd(dap, cycle_days, emergence_gdd,
                             maximum_rooting_gdd, cumulative_gdd,
                             minimum_depth, maximum_depth, shape_factor,
                             delayed_days=0, initial_root_percent=70.0):
    virtual_day = jnp.asarray(dap, jnp.int32) - jnp.asarray(delayed_days, jnp.int32)
    gdd, emergence, maximum_gdd, zmin, zmax = [jnp.asarray(v, F64) for v in
        (cumulative_gdd, emergence_gdd, maximum_rooting_gdd,
         minimum_depth, maximum_depth)]
    initial_depth = zmin * jnp.asarray(initial_root_percent, F64) / 100.0
    initial_gdd = emergence / 2.0
    linear = initial_depth + (zmax - initial_depth) * jnp.where(
        maximum_gdd != 0.0, gdd / maximum_gdd, 0.0)
    shaped = initial_depth + (zmax - initial_depth) * time_root_function(
        gdd, shape_factor, maximum_gdd, initial_gdd)
    developing = jnp.where(maximum_gdd <= initial_gdd, linear,
                           jnp.where(gdd <= initial_gdd, initial_depth, shaped))
    depth = jnp.where(gdd >= maximum_gdd, zmax,
                      jnp.where(zmin < zmax, jnp.maximum(developing, zmin), zmax))
    active = ((virtual_day >= 1) &
              (virtual_day <= jnp.asarray(cycle_days, jnp.int32)))
    return jnp.where(active, depth, jnp.zeros_like(depth))


@jax.jit
def root_zone_water(theta, thickness, gravel_volume_pct, rooting_depth,
                    field_capacity_pct, wilting_point_pct, saturation_pct):
    theta = jnp.asarray(theta, F64)
    thickness = jnp.asarray(thickness, F64)
    gravel = jnp.asarray(gravel_volume_pct, F64) / 100.0
    depth = jnp.maximum(jnp.asarray(rooting_depth, F64), 0.0)
    starts = jnp.concatenate((jnp.zeros((1,), F64), jnp.cumsum(thickness)[:-1]))
    fractions = jnp.clip((depth - starts) / thickness, 0.0, 1.0)
    effective = 1000.0 * thickness * fractions * (1.0 - gravel)
    actual = jnp.sum(theta * effective)
    fc = jnp.sum(jnp.asarray(field_capacity_pct, F64) / 100.0 * effective)
    wp = jnp.sum(jnp.asarray(wilting_point_pct, F64) / 100.0 * effective)
    sat = jnp.sum(jnp.asarray(saturation_pct, F64) / 100.0 * effective)
    available = fc - wp
    relative = jnp.clip(jnp.where(available != 0.0,
                                  (fc - actual) / available, 0.0), 0.0, 1.0)
    return RootZoneWater(actual, fc, wp, sat, fc - actual, available,
                         relative, fractions)
