"""Recurrent coupling of the JAX water and evaporation ports."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .evaporation import stage1_evaporation, stage2_evaporation, water_in_evap_layer
from .hydrology import daily_water_step
from .soil import profile_water_mm, water_balance_residual

F64 = jnp.float64


class WaterEvapTrajectory(NamedTuple):
    theta: jax.Array
    runoff: jax.Array
    drainage: jax.Array
    infiltration: jax.Array
    evaporation: jax.Array
    evap_surface_water: jax.Array
    evap_depth: jax.Array
    stage2_start: jax.Array
    residual: jax.Array


def prepare_stage2_percentage(theta, thickness, gravel_volume_pct,
                              saturation_pct, field_capacity_pct,
                              rew_mm, evap_depth_m=0.15):
    theta = jnp.asarray(theta, F64)
    sat = water_in_evap_layer(jnp.full(theta.shape, jnp.asarray(saturation_pct, F64) / 100.0), thickness, gravel_volume_pct, evap_depth_m)
    fc = water_in_evap_layer(jnp.full(theta.shape, jnp.asarray(field_capacity_pct, F64) / 100.0), thickness, gravel_volume_pct, evap_depth_m)
    actual = water_in_evap_layer(theta, thickness, gravel_volume_pct, evap_depth_m)
    base = fc - jnp.asarray(rew_mm, F64)
    raw = jnp.where(actual - base <= jnp.finfo(F64).eps, 0.0,
                    100.0 * (actual - base) / (sat - base))
    return jnp.floor(raw + 0.5)


def run_water_evap_trajectory(
        initial_theta, rain, eto, thickness, gravel_volume_pct, adjusted_fc,
        saturation_pct, field_capacity_pct, wilting_point_pct, tau, ksat,
        curve_number, rew_mm, bare_soil_coefficient=1.10,
        initial_abstraction_pct=5.0, maximum_evap_depth_m=0.30,
        decline_factor=4.0, initial_surface_water_mm=0.0,
        initial_evap_depth_m=0.15, initial_stage2_start_pct=-1.0):
    initial_theta = jnp.asarray(initial_theta, F64)
    rain = jnp.asarray(rain, F64)
    eto = jnp.asarray(eto, F64)
    initial_depth = jnp.asarray(initial_evap_depth_m, F64)
    calculated = prepare_stage2_percentage(
        initial_theta, thickness, gravel_volume_pct, saturation_pct,
        field_capacity_pct, rew_mm, initial_depth)
    supplied = jnp.asarray(initial_stage2_start_pct, F64)
    initial_start = jnp.where(supplied < 0.0, calculated, supplied)

    def body(carry, forcing):
        current_theta, surface_store, depth, stage2_start = carry
        rain_day, eto_day = forcing
        begin = profile_water_mm(current_theta, thickness, gravel_volume_pct)
        water = daily_water_step(
            current_theta, thickness, gravel_volume_pct, adjusted_fc,
            saturation_pct, field_capacity_pct, tau, ksat, rain_day,
            curve_number, initial_abstraction_pct)
        wetting = jnp.maximum(rain_day - water.runoff, 0.0)
        rew = jnp.asarray(rew_mm, F64)
        is_wet = rain_day > 0.0
        surface_store = jnp.where(is_wet, jnp.minimum(wetting, rew), surface_store)
        depth = jnp.where(is_wet, jnp.asarray(0.15, F64), depth)
        first = stage1_evaporation(
            water.theta, thickness, gravel_volume_pct, wilting_point_pct,
            eto_day * jnp.asarray(bare_soil_coefficient, F64), surface_store, 0.15)
        recalculated = prepare_stage2_percentage(
            first.theta, thickness, gravel_volume_pct, saturation_pct,
            field_capacity_pct, rew, 0.15)
        stage2_start = jnp.where(jnp.logical_and(is_wet,
                                      first.remaining_surface_water < 1e-12),
                                 recalculated, stage2_start)
        second = stage2_evaporation(
            first.theta, thickness, gravel_volume_pct, saturation_pct,
            field_capacity_pct, wilting_point_pct, rew, stage2_start,
            first.remaining_potential, depth, maximum_evap_depth_m,
            decline_factor)
        evaporation = first.actual_evaporation + second.actual_evaporation
        end = profile_water_mm(second.theta, thickness, gravel_volume_pct)
        residual = water_balance_residual(
            begin, 0.0, end, 0.0, rain_day, 0.0, 0.0, 0.0,
            water.drainage, water.runoff, evaporation, 0.0)
        output = (second.theta, water.runoff, water.drainage,
                  water.infiltration, evaporation, first.remaining_surface_water,
                  second.evaporation_depth, stage2_start, residual)
        carry = (second.theta, first.remaining_surface_water,
                 second.evaporation_depth, stage2_start)
        return carry, output

    _, output = lax.scan(
        body,
        (initial_theta, jnp.asarray(initial_surface_water_mm, F64),
         initial_depth, initial_start),
        (rain, eto))
    return WaterEvapTrajectory(*output)
