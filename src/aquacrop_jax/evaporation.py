"""JAX soil-evaporation operators translated from ``simul.f90``."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

F64 = jnp.float64


class ExtractionResult(NamedTuple):
    theta: jax.Array
    extracted: jax.Array


class Stage1Result(NamedTuple):
    theta: jax.Array
    actual_evaporation: jax.Array
    remaining_surface_water: jax.Array
    remaining_potential: jax.Array


class Stage2Result(NamedTuple):
    theta: jax.Array
    actual_evaporation: jax.Array
    evaporation_depth: jax.Array
    remaining_potential: jax.Array


def extract_water_from_evap_layer(theta, thickness, gravel_volume_pct,
                                  wilting_point_pct, demand_mm, depth_m):
    """Extract top-down to AquaCrop's air-dry limit (WP/2)."""
    theta = jnp.asarray(theta, F64)
    thickness = jnp.asarray(thickness, F64)
    gravel = jnp.asarray(gravel_volume_pct, F64) / 100.0
    wp = jnp.asarray(wilting_point_pct, F64)
    demand = jnp.maximum(jnp.asarray(demand_mm, F64), 0.0)
    depth = jnp.asarray(depth_m, F64)

    def body(carry, inputs):
        ztotal, extracted = carry
        water_content, dz, gravel_fraction = inputs
        fraction = jnp.clip((depth - ztotal) / dz, 0.0, 1.0)
        capacity = 1000.0 * dz * (1.0 - gravel_fraction)
        water = water_content * capacity
        air_dry = 10.0 * (wp / 2.0) * dz * (1.0 - gravel_fraction)
        available = jnp.maximum((water - air_dry) * fraction, 0.0)
        active = jnp.logical_and(ztotal < depth * 0.999999,
                                 extracted + 1e-12 < demand)
        removed = jnp.where(active,
                            jnp.minimum(available, demand - extracted), 0.0)
        updated = jnp.where(capacity != 0.0, (water - removed) / capacity,
                            water_content)
        return (ztotal + jnp.where(active, fraction * dz, 0.0),
                extracted + removed), updated

    (_, extracted), result = lax.scan(
        body, (jnp.asarray(0.0, F64), jnp.asarray(0.0, F64)),
        (theta, thickness, gravel))
    return ExtractionResult(result, extracted)


def stage1_evaporation(theta, thickness, gravel_volume_pct, wilting_point_pct,
                       potential_mm, surface_water_mm, evap_depth_m=0.10):
    potential = jnp.maximum(jnp.asarray(potential_mm, F64), 0.0)
    store = jnp.maximum(jnp.asarray(surface_water_mm, F64), 0.0)
    requested = jnp.minimum(potential, store)
    extraction = extract_water_from_evap_layer(
        theta, thickness, gravel_volume_pct, wilting_point_pct,
        requested, evap_depth_m)
    remaining = jnp.maximum(store - extraction.extracted, 0.0)
    remaining = jnp.where(jnp.abs(requested - extraction.extracted) > 0.0001,
                          0.0, remaining)
    return Stage1Result(extraction.theta, extraction.extracted, remaining,
                        jnp.maximum(potential - extraction.extracted, 0.0))


def evaporation_reduction_coefficient(relative_water, decline_factor):
    water = jnp.asarray(relative_water, F64)
    decline = jnp.asarray(decline_factor, F64)
    interior = jnp.expm1(decline * water) / jnp.expm1(decline)
    return jnp.where(water <= 0.00001, jnp.zeros_like(interior),
                     jnp.where(water >= 0.99999, jnp.ones_like(interior),
                               interior))


def water_in_evap_layer(theta, thickness, gravel_volume_pct, depth_m):
    theta = jnp.asarray(theta, F64)
    thickness = jnp.asarray(thickness, F64)
    gravel = jnp.asarray(gravel_volume_pct, F64) / 100.0
    depth = jnp.asarray(depth_m, F64)
    starts = jnp.concatenate((jnp.zeros((1,), F64), jnp.cumsum(thickness)[:-1]))
    fraction = jnp.clip((depth - starts) / thickness, 0.0, 1.0)
    return jnp.sum(1000.0 * theta * thickness * fraction * (1.0 - gravel))


def stage2_evaporation(theta, thickness, gravel_volume_pct,
                       saturation_pct, field_capacity_pct, wilting_point_pct,
                       rew_mm, stage2_start_pct, potential_mm,
                       evaporation_depth_m=0.15, maximum_depth_m=0.30,
                       decline_factor=4.0):
    """Twenty-substep AquaCrop stage-2 evaporation."""
    theta = jnp.asarray(theta, F64)
    thickness = jnp.asarray(thickness, F64)
    gravel = jnp.asarray(gravel_volume_pct, F64)
    potential = jnp.maximum(jnp.asarray(potential_mm, F64), 0.0)
    max_depth = jnp.asarray(maximum_depth_m, F64)
    min_depth = jnp.asarray(0.15, F64)
    start_fraction = jnp.asarray(stage2_start_pct, F64) / 100.0

    def limits(values, depth):
        shape = values.shape
        sat = water_in_evap_layer(jnp.full(shape, jnp.asarray(saturation_pct, F64) / 100.0), thickness, gravel, depth)
        fc = water_in_evap_layer(jnp.full(shape, jnp.asarray(field_capacity_pct, F64) / 100.0), thickness, gravel, depth)
        lower = water_in_evap_layer(jnp.full(shape, jnp.asarray(wilting_point_pct, F64) / 200.0), thickness, gravel, depth)
        upper = start_fraction * (sat - (fc - jnp.asarray(rew_mm, F64))) + (fc - jnp.asarray(rew_mm, F64))
        return upper, lower

    def outer_body(_, carry):
        values, depth, evaporated = carry

        def relative(current_depth):
            upper, lower = limits(values, current_depth)
            actual = water_in_evap_layer(values, thickness, gravel, current_depth)
            return jnp.where(upper != lower, (actual - lower) / (upper - lower), 0.0)

        def expand_condition(current_depth):
            threshold = 0.4 * (max_depth - current_depth) / (max_depth - min_depth)
            return jnp.logical_and(relative(current_depth) < threshold,
                                   current_depth < max_depth)

        depth = lax.while_loop(expand_condition,
                               lambda z: jnp.minimum(z + 0.001, max_depth), depth)
        coefficient = evaporation_reduction_coefficient(relative(depth), decline_factor)
        extraction = extract_water_from_evap_layer(
            values, thickness, gravel, wilting_point_pct,
            coefficient * potential / 20.0, depth)
        return extraction.theta, depth, evaporated + extraction.extracted

    values, depth, evaporated = lax.fori_loop(
        0, 20, outer_body,
        (theta, jnp.asarray(evaporation_depth_m, F64), jnp.asarray(0.0, F64)))
    return Stage2Result(values, evaporated, depth,
                        jnp.maximum(potential - evaporated, 0.0))
