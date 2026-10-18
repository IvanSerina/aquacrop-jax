"""JAX translation of AquaCrop's compartment transpiration kernel."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .roots import root_zone_water
from .salinity import (adjusted_stomatal_stress, root_zone_conductivity,
                       round_fortran)


F64 = jnp.float64


class TranspirationResult(NamedTuple):
    theta: jax.Array
    actual_mm: jax.Array
    potential_after_rootzone_stress_mm: jax.Array
    root_zone_stress: jax.Array
    root_fractions: jax.Array
    sink_demand_mm: jax.Array
    sink_capacity_mm: jax.Array
    extraction_mm: jax.Array


def stress_coefficient(relative_depletion, upper_limit, lower_limit,
                       shape_factor):
    """Exact functional form of global.f90 ``KsAny``."""
    values = [jnp.asarray(v, F64) for v in
              (relative_depletion, upper_limit, lower_limit, shape_factor)]
    depletion, upper, lower, shape = values
    upper = jnp.where(lower - upper < 1e-4, lower - 1e-4, upper)
    relative = (depletion - upper) / (lower - upper)
    effective_shape = jnp.where(jnp.abs(round_fortran(10.0 * shape)) == 0.0,
                                jnp.asarray(0.01, F64), shape)
    # Preserve the source operation order (exp(x)-1); the small difference
    # from expm1 is observable in the unrounded fertility canopy-decline term.
    curved = 1.0 - (jnp.exp(relative * effective_shape) - 1.0) / (
        jnp.exp(effective_shape) - 1.0)
    return jnp.where(relative <= jnp.finfo(F64).eps, jnp.ones_like(curved),
                     jnp.where(relative >= 1.0, jnp.zeros_like(curved),
                               jnp.clip(curved, 0.0, 1.0)))


def root_sink_profile(thickness, rooting_depth, smax_top, smax_bottom,
                      correction=1.0, net_irrigation=False):
    """Translate root fractions and ``calculate_sink_values``."""
    thickness = jnp.asarray(thickness, F64)
    depth = jnp.maximum(jnp.asarray(rooting_depth, F64), 0.0)
    ends = jnp.cumsum(thickness)
    starts = ends - thickness
    fractions = jnp.clip((depth - starts) / thickness, 0.0, 1.0)
    top = jnp.asarray(smax_top, F64)
    bottom = jnp.asarray(smax_bottom, F64)
    corrected_bottom = bottom * jnp.asarray(correction, F64)
    lower_values = jnp.where(
        ends <= depth,
        corrected_bottom + (top - corrected_bottom) *
        jnp.where(depth != 0.0, (depth - ends) / depth, 0.0),
        jnp.full(ends.shape, corrected_bottom))
    upper_values = jnp.concatenate((jnp.reshape(top, (1,)), lower_values[:-1]), 0)
    distributed = jnp.minimum((upper_values + lower_values) / 2.0, 0.06)
    distributed = jnp.where(fractions > 0.0, distributed, jnp.zeros_like(distributed))
    uniform = jnp.full(thickness.shape, (top + bottom) / 2.0)
    sink = jnp.where(jnp.asarray(net_irrigation, jnp.bool_), uniform, distributed)
    return fractions, sink


def compartment_transpiration(
        theta, thickness, gravel_volume_pct, field_capacity_pct,
        wilting_point_pct, saturation_pct, rooting_depth, potential_mm,
        p_stomatal, smax_top, smax_bottom, stomatal_shape=0.0,
        stress_reduction_pct=0.0, sink_correction=1.0,
        net_irrigation=False, salt_conductivity=None,
        salinity_minimum_ece=2.0, salinity_maximum_ece=16.0,
        salinity_response_pct=100.0, salt_stomatal_coefficients=None):
    """Extract transpiration following ``calculate_transpiration``.

    Optional ECsw stress uses explicit salt conductivity and the calibrated
    polynomial coefficients of AdjustedKsStoToECsw. Delayed anaerobic stress
    remains a separate recurrent operator.
    """
    theta = jnp.asarray(theta, F64)
    thickness = jnp.asarray(thickness, F64)
    gravel = jnp.asarray(gravel_volume_pct, F64)
    fc = jnp.asarray(field_capacity_pct, F64) / 100.0
    wp = jnp.asarray(wilting_point_pct, F64) / 100.0
    potential = jnp.maximum(jnp.asarray(potential_mm, F64), 0.0)
    p_stomatal = jnp.asarray(p_stomatal, F64)
    net = jnp.asarray(net_irrigation, jnp.bool_)

    zone = root_zone_water(theta, thickness, gravel, rooting_depth,
                           field_capacity_pct, wilting_point_pct,
                           saturation_pct)
    zone_ks = stress_coefficient(zone.relative_depletion, p_stomatal, 1.0, 0.0)
    fertility_ks = jnp.clip(
        1.0 - jnp.asarray(stress_reduction_pct, F64) / 100.0, 0.0, 1.0)
    zone_ks = jnp.where(zone.actual_mm < 0.999 * (
        zone.field_capacity_mm - p_stomatal * zone.total_available_mm),
                       fertility_ks * zone_ks, fertility_ks)
    if salt_conductivity is not None:
        if salt_stomatal_coefficients is None:
            raise ValueError("ECsw stress requires calibrated salt_stomatal_coefficients")
        salt_zone = root_zone_conductivity(salt_conductivity, thickness, rooting_depth)
        wrel = jnp.where(zone.total_available_mm > 0.0,
                         zone.depletion_mm / zone.total_available_mm, 0.0)
        zone_ks = adjusted_stomatal_stress(
            *salt_zone, wrel, salinity_minimum_ece, salinity_maximum_ece,
            salinity_response_pct, salt_stomatal_coefficients, zone_ks)
    zone_ks = jnp.where(net, jnp.ones_like(zone_ks), zone_ks)
    target = potential * zone_ks

    fractions, smax = root_sink_profile(
        thickness, rooting_depth, smax_top, smax_bottom, sink_correction, net)
    theta_critical = fc - (fc - wp) * p_stomatal
    local_relative = jnp.where(fc != wp, (fc - theta) / (fc - wp), 0.0)
    local_ks = jnp.where(
        theta >= theta_critical, fertility_ks,
        jnp.where(theta > wp,
                 fertility_ks * stress_coefficient(
                     local_relative, p_stomatal, 1.0, stomatal_shape),
                  jnp.zeros_like(theta)))
    if salt_conductivity is not None:
        local_ks = adjusted_stomatal_stress(
            *salt_conductivity, local_relative, salinity_minimum_ece,
            salinity_maximum_ece, salinity_response_pct,
            salt_stomatal_coefficients, local_ks)
    local_ks = jnp.where(net, jnp.ones_like(local_ks), local_ks)
    unconstrained_capacity = 1000.0 * fractions * smax * thickness
    capacity = local_ks * unconstrained_capacity
    total_unconstrained = jnp.sum(jnp.where(
        fractions > 1e-5, unconstrained_capacity,
        jnp.zeros_like(unconstrained_capacity)))
    demand = jnp.where(
        fractions > 1e-5,
        jnp.where(total_unconstrained != 0.0, target / total_unconstrained, 0.0) *
        unconstrained_capacity,
        jnp.zeros_like(unconstrained_capacity))

    def body(carry, inputs):
        residue, extracted = carry
        requested, available_capacity = inputs
        candidate = jnp.minimum(requested + residue, available_capacity)
        new_residue = jnp.maximum(residue + requested - available_capacity, 0.0)
        remaining = jnp.maximum(target - extracted, 0.0)
        amount = jnp.where(jnp.logical_and(requested > 1e-5, remaining > 1e-5),
                           jnp.minimum(candidate, remaining), 0.0)
        return (new_residue, extracted + amount), amount

    (_, actual), extraction = lax.scan(
        body, (jnp.asarray(0.0, F64), jnp.asarray(0.0, F64)),
        (demand, capacity))
    effective_volume = 1000.0 * thickness * (1.0 - gravel / 100.0)
    updated = theta - jnp.where(effective_volume != 0.0,
                                extraction / effective_volume, 0.0)
    return TranspirationResult(updated, actual, target, zone_ks, fractions,
                               demand, capacity, extraction)
