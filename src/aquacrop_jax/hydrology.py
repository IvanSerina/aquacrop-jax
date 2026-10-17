"""Progressive JAX port of the AquaCrop daily water budget."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .kernels import curve_number_runoff
from .soil import profile_water_mm, vertical_drainage, water_balance_residual


class InfiltrationResult(NamedTuple):
    theta: jax.Array
    fluxout: jax.Array
    deep_percolation: jax.Array
    rejected: jax.Array


class WaterStepResult(NamedTuple):
    theta: jax.Array
    runoff: jax.Array
    drainage: jax.Array
    infiltration: jax.Array
    residual: jax.Array
    fluxout: jax.Array


class WaterTrajectory(NamedTuple):
    theta: jax.Array
    runoff: jax.Array
    drainage: jax.Array
    infiltration: jax.Array
    residual: jax.Array

F64 = jnp.float64

@jax.jit
def vertical_infiltration(theta, thickness, gravel_volume_pct, saturation_pct,
                          water_mm, ksat):
    """Move surface water downward through the homogeneous Ottawa profile.

    Each compartment fills to saturation. Downward transfer is capped by
    `ksat`; water rejected by that cap is returned separately for surface
    runoff/storage accounting. Water leaving the last compartment is deep
    percolation.
    """
    theta = jnp.asarray(theta, F64)
    thickness = jnp.asarray(thickness, F64)
    gravel = jnp.asarray(gravel_volume_pct, F64) / 100.0
    saturation = jnp.asarray(saturation_pct, F64) / 100.0
    ksat = jnp.asarray(ksat, F64)
    capacity = 1000.0 * thickness * (1.0 - gravel)
    def body(carry, values):
        incoming, rejected = carry
        current, current_capacity = values
        space = jnp.maximum((saturation - current) * current_capacity, 0.0)
        stored = jnp.minimum(incoming, space)
        remaining = incoming - stored
        outgoing = jnp.minimum(remaining, ksat)
        rejected = rejected + remaining - outgoing
        updated = current + stored / current_capacity
        return (outgoing, rejected), (updated, outgoing)

    (deep, rejected), (theta_out, flux_out) = jax.lax.scan(
        body, (jnp.asarray(water_mm, F64), jnp.asarray(0.0, F64)),
        (theta, capacity))
    return InfiltrationResult(theta_out, flux_out, deep, rejected)


@jax.jit
def daily_water_step(theta, thickness, gravel_volume_pct, adjusted_fc,
                     saturation_pct, field_capacity_pct, tau, ksat, rain,
                     curve_number, initial_abstraction_pct=5.0):
    """Ottawa-path ordering: drainage, runoff, then rainfall infiltration."""
    theta = jnp.asarray(theta, F64)
    thickness = jnp.asarray(thickness, F64)
    gravel = jnp.asarray(gravel_volume_pct, F64)
    begin = profile_water_mm(theta, thickness, gravel)
    drained_theta, drainage_flux, initial_drain = vertical_drainage(
        theta, thickness, gravel, adjusted_fc, saturation_pct,
        field_capacity_pct, tau, ksat)
    runoff_scs = curve_number_runoff(rain, curve_number, initial_abstraction_pct)
    available = jnp.maximum(jnp.asarray(rain, F64) - runoff_scs, 0.0)
    infiltrated = vertical_infiltration(
        drained_theta, thickness, gravel, saturation_pct, available, ksat)
    runoff = runoff_scs + infiltrated.rejected
    drainage = initial_drain + infiltrated.deep_percolation
    actual_infiltration = available - infiltrated.rejected
    end = profile_water_mm(infiltrated.theta, thickness, gravel)
    residual = water_balance_residual(
        begin, 0.0, end, 0.0, rain, 0.0, 0.0, 0.0,
        drainage, runoff, 0.0, 0.0)
    return WaterStepResult(infiltrated.theta, runoff, drainage,
                           actual_infiltration, residual,
                           drainage_flux + infiltrated.fluxout)


@jax.jit
def daily_water_step_irrigated(theta, thickness, gravel_volume_pct,
                               adjusted_fc, saturation_pct,
                               field_capacity_pct, tau, ksat, rain,
                               irrigation, curve_number,
                               initial_abstraction_pct=5.0):
    """Drain, calculate rainfall runoff, then infiltrate rain plus irrigation."""
    theta = jnp.asarray(theta, F64)
    rain = jnp.asarray(rain, F64)
    irrigation = jnp.maximum(jnp.asarray(irrigation, F64), 0.0)
    begin = profile_water_mm(theta, thickness, gravel_volume_pct)
    drained_theta, drainage_flux, initial_drain = vertical_drainage(
        theta, thickness, gravel_volume_pct, adjusted_fc, saturation_pct,
        field_capacity_pct, tau, ksat)
    runoff_scs = curve_number_runoff(rain, curve_number, initial_abstraction_pct)
    available = jnp.maximum(rain - runoff_scs, 0.0) + irrigation
    infiltrated = vertical_infiltration(
        drained_theta, thickness, gravel_volume_pct, saturation_pct,
        available, ksat)
    runoff = runoff_scs + infiltrated.rejected
    drainage = initial_drain + infiltrated.deep_percolation
    actual_infiltration = available - infiltrated.rejected
    end = profile_water_mm(infiltrated.theta, thickness, gravel_volume_pct)
    residual = water_balance_residual(
        begin, 0.0, end, 0.0, rain, irrigation, 0.0, 0.0,
        drainage, runoff, 0.0, 0.0)
    return WaterStepResult(infiltrated.theta, runoff, drainage,
                           actual_infiltration, residual,
                           drainage_flux + infiltrated.fluxout)


@jax.jit
def run_water_trajectory(initial_theta, rain, thickness, gravel_volume_pct,
                         adjusted_fc, saturation_pct, field_capacity_pct,
                         tau, ksat, curve_number,
                         initial_abstraction_pct=5.0):
    """Unroll the current water-only daily recurrence over a rain series."""
    initial_theta = jnp.asarray(initial_theta, F64)
    rain = jnp.asarray(rain, F64)

    def body(current_theta, daily_rain):
        result = daily_water_step(
            current_theta, thickness, gravel_volume_pct, adjusted_fc,
            saturation_pct, field_capacity_pct, tau, ksat, daily_rain,
            curve_number, initial_abstraction_pct)
        output = WaterTrajectory(result.theta, result.runoff, result.drainage,
                                 result.infiltration, result.residual)
        return result.theta, output

    _, history = jax.lax.scan(body, initial_theta, rain)
    return history
