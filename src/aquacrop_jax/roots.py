"""JAX root-depth and root-zone water operators."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .rounding import round_fortran

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


class CanopyWaterZone(NamedTuple):
    """Water state chosen by ``DetermineRootZoneWC`` for canopy stress."""
    actual_mm: jax.Array
    field_capacity_mm: jax.Array
    wilting_point_mm: jax.Array
    topsoil_considered: jax.Array


def canopy_water_zone(theta, thickness, gravel_volume_pct, rooting_depth,
                      field_capacity_pct, wilting_point_pct, saturation_pct,
                      topsoil_depth_m):
    """Choose topsoil only if it is relatively wetter than the root zone."""
    root = root_zone_water(theta, thickness, gravel_volume_pct, rooting_depth,
                           field_capacity_pct, wilting_point_pct,
                           saturation_pct)
    top_depth = jnp.minimum(jnp.asarray(rooting_depth, F64),
                            jnp.asarray(topsoil_depth_m, F64))
    top = root_zone_water(theta, thickness, gravel_volume_pct, top_depth,
                          field_capacity_pct, wilting_point_pct,
                          saturation_pct)

    def relative(zone):
        available = zone.field_capacity_mm - zone.wilting_point_mm
        return jnp.where(round_fortran(1000.0 * available) > 0.0,
                         (zone.field_capacity_mm - zone.actual_mm) / available,
                         0.0)

    wetter_top = relative(top) < relative(root)
    return CanopyWaterZone(
        jnp.where(wetter_top, top.actual_mm, root.actual_mm),
        jnp.where(wetter_top, top.field_capacity_mm, root.field_capacity_mm),
        jnp.where(wetter_top, top.wilting_point_mm, root.wilting_point_mm),
        wetter_top)


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
def actual_rooting_depth_calendar(dap, cycle_days, emergence_days,
                                  maximum_rooting_days, minimum_depth,
                                  maximum_depth, shape_factor,
                                  delayed_days=0, initial_root_percent=70.0):
    """ActualRootingDepthDays from ``global.f90``.

    ``dap`` is one-based at the end of the simulated day.  The initial
    rooting depth is present from day one, while expansion is held at
    ``Zmin * RootPercentZmin`` through half the emergence period and then
    follows the same shaped time function used by the native routine.
    """
    virtual_day = jnp.asarray(dap, jnp.int32) - jnp.asarray(delayed_days, jnp.int32)
    cycle, emergence, maximum_days = [jnp.asarray(v, jnp.int32) for v in
                                      (cycle_days, emergence_days, maximum_rooting_days)]
    zmin, zmax, shape = [jnp.asarray(v, F64) for v in
                          (minimum_depth, maximum_depth, shape_factor)]
    zini = zmin * jnp.asarray(initial_root_percent, F64) / 100.0
    t0 = round_fortran(jnp.asarray(emergence, F64) / 2.0).astype(jnp.int32)
    day = jnp.asarray(virtual_day, F64)
    shaped = zini + (zmax - zini) * time_root_function(
        day, shape, jnp.asarray(maximum_days, F64), jnp.asarray(t0, F64))
    linear = zini + (zmax - zini) * day / jnp.asarray(maximum_days, F64)
    developing = jnp.where(maximum_days <= t0, linear,
                           jnp.where(virtual_day <= t0, zini, shaped))
    depth = jnp.where(virtual_day >= maximum_days, zmax,
                      jnp.where(zmin < zmax, jnp.maximum(developing, zmin), zmax))
    active = (virtual_day >= 1) & (virtual_day <= cycle)
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
