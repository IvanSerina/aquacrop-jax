"""Progressive JAX port of the AquaCrop daily water budget."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .kernels import curve_number_runoff
from .soil import (drainage_delta, profile_water_mm, theta_from_drainage,
                   vertical_drainage, water_balance_residual)


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
    """Simple saturation-fill primitive for isolated diagnostics.

    Each compartment fills to saturation. Downward transfer is capped by
    `ksat`; water rejected by that cap is returned separately for surface
    runoff/storage accounting. Water leaving the last compartment is deep
    percolation. The daily runner uses ``daily_infiltration`` to account for
    field capacity, drainage kinetics and existing interface fluxes.
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
def daily_infiltration(theta, thickness, gravel_volume_pct, adjusted_fc,
                       saturation_pct, field_capacity_pct, tau, ksat,
                       water_mm, initial_fluxout):
    """Native daily Extra_runoff + infiltration, homogeneous soil, no bunds.

    ``fluxout`` includes pre-infiltration drainage. ``deep_percolation`` is
    only the additional bottom flux caused by this surface input.
    """
    theta = jnp.asarray(theta, F64)
    thickness = jnp.asarray(thickness, F64)
    gravel = jnp.asarray(gravel_volume_pct, F64) / 100.0
    capacity = 1000.0 * thickness * (1.0 - gravel)
    adjusted_fc = jnp.asarray(adjusted_fc, F64)
    saturation = jnp.asarray(saturation_pct, F64)
    theta_sat = saturation / 100.0
    fc = jnp.asarray(field_capacity_pct, F64)
    theta_fc = fc / 100.0
    tau = jnp.asarray(tau, F64)
    ksat = jnp.asarray(ksat, F64)
    water = jnp.maximum(jnp.asarray(water_mm, F64), 0.0)
    flux = jnp.asarray(initial_fluxout, F64)
    delta_sat = drainage_delta(theta_sat, theta_fc, saturation, fc, tau)
    eps = jnp.finfo(F64).eps

    def body(index, state):
        def active(state):
            profile, flux, remaining, rejected = state
            factor = jnp.where(delta_sat > 0.0,
                               ksat / (jnp.where(delta_sat > 0.0, delta_sat, 1.0) * capacity[index]), 1.0)
            delta_nul = remaining / capacity[index]

            def below_saturation(_):
                theta_nul = theta_from_drainage(delta_nul, theta_fc, saturation, fc, tau)
                use_adjusted = theta_nul <= adjusted_fc[index]
                theta_nul = jnp.where(use_adjusted, adjusted_fc[index], theta_nul)
                delta = jnp.where(use_adjusted,
                    drainage_delta(theta_nul, theta_fc, saturation, fc, tau), delta_nul)
                return jnp.minimum(theta_nul, theta_sat), delta

            theta_nul, delta_nul = jax.lax.cond(delta_nul < delta_sat, below_saturation,
                lambda _: (theta_sat, delta_sat), operand=None)
            drain_max = jnp.minimum(factor * delta_nul * capacity[index], ksat - flux[index])
            current = profile[index]
            filled = current + remaining / capacity[index]
            store = theta_nul > current
            remaining = jnp.where(store, jnp.maximum((filled - theta_nul) * capacity[index], 0.0), remaining)
            profile = profile.at[index].set(jnp.where(store, jnp.minimum(filled, theta_nul), current))
            flux = flux.at[index].add(remaining)
            excess = jnp.maximum(remaining - drain_max, 0.0)
            remaining -= excess

            def redistribute(offset, carry):
                def move(carry):
                    profile, flux, excess, _ = carry
                    previous = index - offset
                    flux = flux.at[previous].add(-excess)
                    filled = profile[previous] + excess / capacity[previous]
                    excess = jnp.maximum((filled - theta_sat) * capacity[previous], 0.0)
                    profile = profile.at[previous].set(jnp.minimum(filled, theta_sat))
                    return profile, flux, excess, excess >= eps
                return jax.lax.cond(jnp.logical_and(offset <= index, carry[3]),
                                    move, lambda values: values, carry)

            profile, flux, excess, _ = jax.lax.fori_loop(0, theta.shape[0], redistribute,
                (profile, flux, excess, excess > 0.0))
            return profile, flux, remaining, rejected + excess

        threshold = jnp.where(index == 0, 0.0, eps)
        return jax.lax.cond(state[2] > threshold, active, lambda values: values, state)

    surface_input = jnp.minimum(water, ksat)
    profile, flux, deep, rejected = jax.lax.fori_loop(0, theta.shape[0], body,
        (theta, flux, surface_input, water - surface_input))
    return InfiltrationResult(profile, flux, deep, rejected)


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
    infiltrated = daily_infiltration(
        drained_theta, thickness, gravel, adjusted_fc, saturation_pct,
        field_capacity_pct, tau, ksat, available, drainage_flux)
    runoff = runoff_scs + infiltrated.rejected
    drainage = initial_drain + infiltrated.deep_percolation
    actual_infiltration = available - infiltrated.rejected
    end = profile_water_mm(infiltrated.theta, thickness, gravel)
    residual = water_balance_residual(
        begin, 0.0, end, 0.0, rain, 0.0, 0.0, 0.0,
        drainage, runoff, 0.0, 0.0)
    return WaterStepResult(infiltrated.theta, runoff, drainage,
                           actual_infiltration, residual,
                           infiltrated.fluxout)


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
    infiltrated = daily_infiltration(
        drained_theta, thickness, gravel_volume_pct, adjusted_fc, saturation_pct,
        field_capacity_pct, tau, ksat, available, drainage_flux)
    runoff = runoff_scs + infiltrated.rejected
    drainage = initial_drain + infiltrated.deep_percolation
    actual_infiltration = available - infiltrated.rejected
    end = profile_water_mm(infiltrated.theta, thickness, gravel_volume_pct)
    residual = water_balance_residual(
        begin, 0.0, end, 0.0, rain, irrigation, 0.0, 0.0,
        drainage, runoff, 0.0, 0.0)
    return WaterStepResult(infiltrated.theta, runoff, drainage,
                           actual_infiltration, residual,
                           infiltrated.fluxout)


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
