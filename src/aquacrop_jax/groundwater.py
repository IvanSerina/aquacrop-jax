"""Groundwater interpolation and profile adjustment operators."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

F64 = jnp.float64


class GroundwaterState(NamedTuple):
    depth_m: jax.Array
    ec_dsm: jax.Array


def interpolate_groundwater(day_number, observation_days, depth_m, ec_dsm):
    """Piecewise-linear groundwater depth/EC, constant beyond endpoints."""
    days = jnp.asarray(observation_days, F64)
    depths = jnp.asarray(depth_m, F64)
    ecs = jnp.asarray(ec_dsm, F64)
    if days.ndim != 1 or days.shape != depths.shape or days.shape != ecs.shape:
        raise ValueError("groundwater observations must be equal one-dimensional arrays")
    if days.shape[0] == 0:
        return GroundwaterState(jnp.asarray(-1.0, F64), jnp.asarray(-1.0, F64))
    day = jnp.asarray(day_number, F64)
    right = jnp.clip(jnp.searchsorted(days, day, side="right"), 1, days.shape[0] - 1)
    left = right - 1
    span = days[right] - days[left]
    fraction = jnp.where(span != 0.0, (day - days[left]) / span, 0.0)
    depth = depths[left] + fraction * (depths[right] - depths[left])
    ec = ecs[left] + fraction * (ecs[right] - ecs[left])
    depth = jnp.where(day <= days[0], depths[0],
                      jnp.where(day >= days[-1], depths[-1], depth))
    ec = jnp.where(day <= days[0], ecs[0],
                   jnp.where(day >= days[-1], ecs[-1], ec))
    return GroundwaterState(depth, ec)


def no_adjustment_distance(field_capacity_pct):
    """Nested ``NoAdjustment`` function from ``CalculateAdjustedFC``."""
    fc = jnp.asarray(field_capacity_pct, F64)
    pf = 2.0 + 0.3 * (fc - 10.0) / 20.0
    middle = jnp.power(10.0, pf) / 100.0
    return jnp.where(fc <= 10.0, 1.0, jnp.where(fc >= 30.0, 2.0, middle))


def adjusted_field_capacity(thickness, field_capacity_pct, saturation_pct,
                            groundwater_depth_m):
    """Vector translation of AquaCrop ``CalculateAdjustedFC``."""
    thickness = jnp.asarray(thickness, F64)
    fc = jnp.asarray(field_capacity_pct, F64)
    sat = jnp.asarray(saturation_pct, F64)
    depth = jnp.asarray(groundwater_depth_m, F64)
    centers = jnp.cumsum(thickness) - thickness / 2.0
    distance = no_adjustment_distance(fc)
    delta = sat - fc
    within = depth - centers < distance
    quadratic = fc + delta / (distance ** 2) * (
        centers - (depth - distance)) ** 2
    adjusted = jnp.where(centers >= depth, sat,
                         jnp.where(within, quadratic, fc))
    adjusted = jnp.where(jnp.logical_or(depth < 0.0, fc >= sat), fc, adjusted)
    return jnp.clip(adjusted, fc, sat)


def saturate_below_water_table(theta, thickness, saturation_pct,
                               groundwater_depth_m):
    """Apply ``AdjustForWatertable`` to compartment water content."""
    theta = jnp.asarray(theta, F64)
    thickness = jnp.asarray(thickness, F64)
    sat = jnp.asarray(saturation_pct, F64) / 100.0
    depth = jnp.asarray(groundwater_depth_m, F64)
    centers = jnp.cumsum(thickness) - thickness / 2.0
    return jnp.where(jnp.logical_and(depth >= 0.0, centers >= depth), sat, theta)


def constrain_root_depth(rooting_depth_m, groundwater_depth_m,
                         minimum_root_depth_m, anaerobic_sensitive=True):
    root = jnp.asarray(rooting_depth_m, F64)
    groundwater = jnp.asarray(groundwater_depth_m, F64)
    minimum = jnp.asarray(minimum_root_depth_m, F64)
    limited = jnp.maximum(minimum, jnp.minimum(root, groundwater))
    active = jnp.logical_and(jnp.asarray(anaerobic_sensitive, jnp.bool_),
                             jnp.logical_and(groundwater >= 0.0,
                                             root > groundwater))
    return jnp.where(active, limited, root)
