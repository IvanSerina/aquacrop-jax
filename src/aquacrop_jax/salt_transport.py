"""Salt advection and diffusion from simul.f90 calculate_saltcontent.

This daily-input path supports explicit compartment fluxes. SubDrain for
monthly/decadal precipitation is not part of this daily forcing interface.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .salinity import (EQUIV, F64, MAX_CELLS, SaltGeometry, SaltState,
                       active_cells, compartment_conductivity, solution_deposit)


class SaltStep(NamedTuple):
    state: SaltState
    infiltrated_g_m2: jax.Array
    drained_g_m2: jax.Array
    drainage_ec: jax.Array
    residual_g_m2: jax.Array


def mix_cells(mobility, water1, water2, salt1, salt2, deposit1, deposit2, solubility=100.0):
    first = solution_deposit(water1, salt1, deposit1, solubility)
    second = solution_deposit(water2, salt2, deposit2, solubility)
    mixed = (first.dissolved + second.dissolved) / (water1 + water2)
    salt1 = first.dissolved + (mixed * water1 - first.dissolved) * mobility
    salt2 = second.dissolved + (mixed * water2 - second.dissolved) * mobility
    return (solution_deposit(water1, salt1, first.deposited, solubility),
            solution_deposit(water2, salt2, second.deposited, solubility))


def _move_salt(solution, deposit, cell, delta, count, water_cell, solubility):
    """MoveSaltTo: withdrawals start at the last cell and borrow downward."""
    start = jnp.where(delta >= jnp.finfo(F64).eps, cell, count - 1)
    solution = solution.at[start].add(delta)
    mm = jnp.where(start == count - 1, 2.0 * water_cell, water_cell)
    balanced = solution_deposit(mm, solution[start], deposit[start], solubility)
    solution = solution.at[start].set(balanced.dissolved)
    deposit = deposit.at[start].set(balanced.deposited)

    def body(index, carry):
        salt, depo = carry
        current = count - 1 - index
        current = jnp.clip(current, 1, MAX_CELLS - 1)
        borrow = (delta < jnp.finfo(F64).eps) & (index < count - 1) & (salt[current] < 0.0)
        amount = jnp.where(borrow, salt[current], 0.0)
        salt = salt.at[current].add(-amount).at[current - 1].add(amount)
        balanced = solution_deposit(water_cell, salt[current - 1], depo[current - 1], solubility)
        salt = salt.at[current - 1].set(jnp.where(borrow, balanced.dissolved, salt[current - 1]))
        depo = depo.at[current - 1].set(jnp.where(borrow, balanced.deposited, depo[current - 1]))
        return salt, depo

    return lax.fori_loop(0, MAX_CELLS - 1, body, (solution, deposit))


def salt_transport_step(state, theta_after_flow, fluxout, thickness,
                         gravel_volume_pct, saturation_pct, field_capacity_pct,
                         geometry, infiltrated_rain=0.0, infiltrated_irrigation=0.0,
                         irrigation_ec=0.0, infiltrated_storage=0.0, storage_ec=0.0,
                         solubility=100.0):
    theta = jnp.asarray(theta_after_flow, F64)
    flux = jnp.asarray(fluxout, F64)
    thickness = jnp.asarray(thickness, F64)
    volume = 1000.0 * thickness * (1.0 - jnp.asarray(gravel_volume_pct, F64) / 100.0)
    sat = jnp.broadcast_to(jnp.asarray(saturation_pct, F64) / 100.0, theta.shape)
    initial_mass = jnp.sum(state.dissolved + state.deposited)
    incoming_water = jnp.asarray(infiltrated_rain, F64) + infiltrated_irrigation + infiltrated_storage
    incoming_salt = EQUIV * (jnp.asarray(infiltrated_irrigation, F64) * irrigation_ec
                              + jnp.asarray(infiltrated_storage, F64) * storage_ec)

    def compartment(carry, inputs):
        water_in, salt_in = carry
        solution, deposit, final_theta, outflow, dz, vol, saturation, count, upper, dx, mobility = inputs
        # Preserve the reference's gravel-independent reconstruction term.
        wet_theta = final_theta + outflow / (1000.0 * dz)
        local_geometry = SaltGeometry(count, upper, dx, mobility)
        active = active_cells(wet_theta, local_geometry)
        solution = solution.at[active].add(jnp.where(water_in > 0.0, salt_in, 0.0))

        def mixing(index, stores):
            salts, deposits = stores
            mm1 = dx * vol
            mm2 = jnp.where(index < count - 2, mm1,
                            (jnp.maximum(wet_theta, saturation) - upper) * vol)
            first, second = mix_cells(mobility[index], mm1, mm2,
                salts[index], salts[index + 1], deposits[index], deposits[index + 1], solubility)
            enabled = index < active
            salts = salts.at[index].set(jnp.where(enabled, first.dissolved, salts[index]))
            salts = salts.at[index + 1].set(jnp.where(enabled, second.dissolved, salts[index + 1]))
            deposits = deposits.at[index].set(jnp.where(enabled, first.deposited, deposits[index]))
            deposits = deposits.at[index + 1].set(jnp.where(enabled, second.deposited, deposits[index + 1]))
            return salts, deposits

        solution, deposit = lax.fori_loop(0, MAX_CELLS - 1, mixing, (solution, deposit))

        def drain(_, carry):
            salts, deposits, wet, delta, cell, salt_out = carry
            cell = jnp.maximum(cell, 0)
            limit = jnp.where(cell < count - 1, cell * dx, upper)
            span = wet - limit
            full = wet - delta < limit
            enabled = (delta > 0.0) & (span > 0.0)
            fraction = jnp.minimum(delta / jnp.where(span > 0.0, span, 1.0), 1.0)
            removed = (salts[cell] + deposits[cell]) * fraction
            candidate = salt_out + removed
            mm = jnp.minimum(delta, jnp.maximum(span, 0.0)) * vol
            excess = jnp.maximum(candidate - solubility * mm, 0.0)
            new_salt = salts[cell] * (1.0 - fraction)
            new_deposit = deposits[cell] * (1.0 - fraction) + excess
            cell_water = jnp.where(cell == count - 1, 2.0, 1.0) * dx * vol
            balanced = solution_deposit(cell_water, new_salt, new_deposit, solubility)
            new_salt = jnp.where(full, 0.0, balanced.dissolved)
            new_deposit = jnp.where(full, excess, balanced.deposited)
            salts = salts.at[cell].set(jnp.where(enabled, new_salt, salts[cell]))
            deposits = deposits.at[cell].set(jnp.where(enabled, new_deposit, deposits[cell]))
            return (salts, deposits, jnp.where(enabled & full, limit, wet),
                    jnp.where(enabled, jnp.where(full, delta - span, 0.0), delta),
                    jnp.where(enabled & full, cell - 1, cell),
                    jnp.where(enabled, candidate - excess, salt_out))

        solution, deposit, _, _, _, salt_out = lax.fori_loop(
            0, MAX_CELLS + 1, drain,
            (solution, deposit, wet_theta, outflow / vol, active, jnp.asarray(0.0, F64)))
        return (outflow, salt_out), (solution, deposit)

    (_, salt_out), (solution, deposit) = lax.scan(compartment, (incoming_water, incoming_salt),
        (state.dissolved, state.deposited, theta, flux, thickness, volume,
         sat, *geometry))
    # Vertical diffusion uses conductivities cached before the neighbor sweep.
    ec = compartment_conductivity(SaltState(solution, deposit), theta, thickness,
        gravel_volume_pct, saturation_pct, field_capacity_pct, solubility).ecsw
    active = active_cells(theta, geometry)
    mobility = jnp.take_along_axis(geometry.mobility, active[:, None], axis=1)[:, 0] / 4.0
    water = theta * volume

    def diffuse(index, stores):
        salts, deposits = stores
        left, right = index - 1, index
        water_total = water[left] + water[right]
        mixed = (ec[left] * water[left] + ec[right] * water[right]) / jnp.where(water_total > 0.0, water_total, 1.0)
        ds1 = (ec[left] - mixed) * mobility[left] * water[left] * EQUIV
        ds2 = (ec[right] - mixed) * mobility[right] * water[right] * EQUIV
        delta = jnp.minimum(jnp.abs(ds1), jnp.abs(ds2)) * jnp.where(ec[left] > mixed, -1.0, 1.0)
        sl, dl = _move_salt(salts[left], deposits[left], active[left], delta,
                            geometry.count[left], geometry.width[left] * volume[left], solubility)
        sr, dr = _move_salt(salts[right], deposits[right], active[right], -delta,
                            geometry.count[right], geometry.width[right] * volume[right], solubility)
        enabled = jnp.abs(delta) > 0.0
        salts = salts.at[left].set(jnp.where(enabled, sl, salts[left]))
        salts = salts.at[right].set(jnp.where(enabled, sr, salts[right]))
        deposits = deposits.at[left].set(jnp.where(enabled, dl, deposits[left]))
        deposits = deposits.at[right].set(jnp.where(enabled, dr, deposits[right]))
        return salts, deposits

    solution, deposit = lax.fori_loop(1, theta.shape[0], diffuse, (solution, deposit))
    final_mass = jnp.sum(solution + deposit)
    drainage_ec = jnp.where(flux[-1] > 0.001, salt_out / (jnp.maximum(flux[-1], 1e-30) * EQUIV), 0.0)
    return SaltStep(SaltState(solution, deposit), incoming_salt, salt_out,
                    drainage_ec, initial_mass + incoming_salt - salt_out - final_mass)
