"""Capillary-rise water flux translated from ``calculate_CapillaryRise``."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

F64 = jnp.float64


class CapillaryRiseResult(NamedTuple):
    theta: jax.Array
    water_mm: jax.Array
    compartment_mm: jax.Array


def maximum_capillary_rise(param_a, param_b, ksat, compartment_center_m,
                           groundwater_depth_m):
    """Exact ``MaxCRatDepth`` guard and exponential relation."""
    a, b, conductivity, center, depth = [jnp.asarray(v, F64) for v in
        (param_a, param_b, ksat, compartment_center_m, groundwater_depth_m)]
    separation = depth - center
    calculated = jnp.exp((jnp.log(jnp.maximum(separation, 1e-300)) - b) / a)
    value = jnp.where(center >= depth, 99.0, jnp.minimum(calculated, 99.0))
    valid = jnp.logical_and(conductivity > 0.0,
            jnp.logical_and(depth > 0.0, separation < 4.0))
    return jnp.where(valid, value, 0.0)


def capillary_rise(theta, thickness, gravel_volume_pct, adjusted_fc_pct,
                   field_capacity_pct, wilting_point_pct, ksat,
                   capillary_a, capillary_b, groundwater_depth_m,
                   fluxout=None, root_shape_factor=16.0):
    """Move groundwater upward through compartments from bottom to top."""
    theta = jnp.asarray(theta, F64)
    thickness = jnp.asarray(thickness, F64)
    gravel = jnp.asarray(gravel_volume_pct, F64) / 100.0
    adjusted_fc = jnp.asarray(adjusted_fc_pct, F64) / 100.0
    fc = jnp.broadcast_to(jnp.asarray(field_capacity_pct, F64) / 100.0,
                          theta.shape)
    wp = jnp.broadcast_to(jnp.asarray(wilting_point_pct, F64) / 100.0,
                          theta.shape)
    ksat = jnp.broadcast_to(jnp.asarray(ksat, F64), theta.shape)
    param_a = jnp.broadcast_to(jnp.asarray(capillary_a, F64), theta.shape)
    param_b = jnp.broadcast_to(jnp.asarray(capillary_b, F64), theta.shape)
    depth = jnp.asarray(groundwater_depth_m, F64)
    shape = jnp.asarray(root_shape_factor, F64)
    outgoing = (jnp.zeros_like(theta) if fluxout is None else
                jnp.asarray(fluxout, F64))
    centers = jnp.cumsum(thickness) - thickness / 2.0

    inputs = tuple(jnp.flip(v) for v in
                   (theta, thickness, gravel, adjusted_fc, fc, wp, ksat,
                    param_a, param_b, centers, outgoing))

    def body(carry, values):
        remaining, total, active = carry
        (water, dz, gravel_fraction, fc_adj, layer_fc, layer_wp,
         conductivity, a, b, center, flux) = values
        local_limit = maximum_capillary_rise(a, b, conductivity, center, depth)
        remaining = jnp.minimum(remaining, local_limit)
        active = jnp.logical_and(active,
            jnp.logical_and(jnp.floor(remaining * 1000.0 + 0.5) > 0.0,
                            jnp.floor(jnp.abs(flux) * 1000.0 + 0.5) == 0.0))
        denominator = fc_adj - layer_wp
        driving = jnp.where(
            jnp.logical_and(water >= layer_wp,
                            jnp.logical_and(shape > 0.0, denominator > 0.0)),
            1.0 - jnp.power(jnp.maximum(water - layer_wp, 0.0) / denominator,
                            shape), 1.0)
        theta_threshold = (layer_wp + layer_fc) / 2.0
        krel = jnp.where(
            water < theta_threshold,
            jnp.where(jnp.logical_or(water <= layer_wp,
                                     theta_threshold <= layer_wp), 0.0,
                      (water - layer_wp) / (theta_threshold - layer_wp)), 1.0)
        room = fc_adj - water
        can_store = jnp.logical_and(active,
            jnp.logical_and(room > 0.0, center < depth))
        maximum_delta = krel * driving * remaining / (1000.0 * dz)
        delta = jnp.where(can_store, jnp.minimum(room, maximum_delta), 0.0)
        stored = delta * 1000.0 * dz * (1.0 - gravel_fraction)
        filled = jnp.logical_and(can_store, room < maximum_delta)
        remaining = jnp.where(
            can_store, jnp.where(filled, jnp.maximum(krel * remaining - stored, 0.0), 0.0),
            remaining)
        updated = water + delta
        return (remaining, total + stored, active), (updated, stored)

    (_, total, _), (updated_reverse, stored_reverse) = lax.scan(
        body, (jnp.asarray(jnp.inf, F64), jnp.asarray(0.0, F64),
               jnp.asarray(True)), inputs)
    return CapillaryRiseResult(jnp.flip(updated_reverse), total,
                               jnp.flip(stored_reverse))
