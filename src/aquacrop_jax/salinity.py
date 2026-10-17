"""AquaCrop 7.3 salt cells, conductivity and stomatal response.

Salt stores are g/m2; water volumes are mm (= l/m2), EC is dS/m.
Equations follow global.f90, including the legacy initial-salt gravel rule.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .canopy import temperature_stress
from .rounding import round_fortran

F64 = jnp.float64
EQUIV = 0.64
MAX_CELLS = 11


class SaltState(NamedTuple):
    dissolved: jax.Array
    deposited: jax.Array


class SaltGeometry(NamedTuple):
    count: jax.Array
    upper: jax.Array
    width: jax.Array
    mobility: jax.Array


class Conductivity(NamedTuple):
    ece: jax.Array
    ecsw: jax.Array
    ecsw_fc: jax.Array


def salt_geometry(saturation_pct, field_capacity_pct, ksat, diffusion_pct=20.0):
    sat, fc, conductivity = jnp.broadcast_arrays(*[
        jnp.atleast_1d(jnp.asarray(x, F64)) for x in
        (saturation_pct, field_capacity_pct, ksat)])
    count = jnp.where(conductivity <= 112.0, 11,
                      jnp.maximum(round_fortran(1.6 + 1000.0 /
                                  jnp.maximum(conductivity, 1e-30)), 2)).astype(jnp.int32)
    sc = count - 1
    upper = sat / 100.0 * sc / (sc + 2.0)
    width = upper / sc
    macro = round_fortran(fc)
    celmax = jnp.maximum(jnp.where(macro > upper * 100.0, count,
        round_fortran(macro / (upper * 100.0) * sc)), 1)
    mix = jnp.asarray(diffusion_pct, F64) / 100.0
    a = jnp.where(mix < 0.5, 2.0 * mix, 2.0 * (1.0 - mix))
    b = 10.0 ** (10.0 * jnp.abs(mix - 0.5))
    cell = jnp.arange(1, MAX_CELLS + 1)
    x = cell / jnp.maximum(celmax[..., None] - 1.0, 1.0)
    denominator = jnp.where(jnp.abs(a * (b - 1.0)) > 1e-30, a * (b - 1.0), 1.0)
    low = (a * b ** x - a) / denominator
    high = 1.0 - (a * b ** (1.0 - x) - a) / denominator
    curved = jnp.where(mix < 0.5, low, high)
    curved = jnp.where(mix == 0.5, x, curved)
    curved = jnp.where(mix <= 0.0, 0.0, jnp.where(mix >= 1.0, 1.0, curved))
    mobility = jnp.where(cell >= celmax[..., None], 1.0, curved)
    mobility = jnp.where(cell <= count[..., None], mobility, 0.0)
    return SaltGeometry(count, upper, width, mobility)


def active_cells(theta, geometry):
    """Return zero-based active-cell indices (global.f90 ActiveCells)."""
    theta = jnp.asarray(theta, F64)
    # Counting comparisons preserves the reference's strict > at cell edges.
    cells = jnp.sum(theta[..., None] > geometry.width[..., None] *
                    jnp.arange(1, MAX_CELLS + 1), axis=-1)
    return jnp.where(theta > geometry.upper, geometry.count - 1,
                     jnp.minimum(cells, geometry.count - 1)).astype(jnp.int32)


def solution_deposit(water_mm, dissolved, deposited, solubility=100.0):
    total = jnp.asarray(dissolved, F64) + jnp.asarray(deposited, F64)
    capacity = jnp.asarray(solubility, F64) * jnp.maximum(jnp.asarray(water_mm, F64), 0.0)
    solution = jnp.minimum(total, capacity)
    return SaltState(solution, total - solution)


def initialize_salt(ece, theta, thickness, gravel_volume_pct, saturation_pct,
                    geometry, solubility=100.0):
    thickness = jnp.asarray(thickness, F64)
    volume = thickness * 1000.0 * (1.0 - jnp.asarray(gravel_volume_pct, F64) / 100.0)
    # DetermineSaltContent intentionally omits gravel from the total initial mass.
    total = jnp.asarray(ece, F64) * EQUIV * jnp.asarray(saturation_pct, F64) * 10.0 * thickness
    active = active_cells(theta, geometry)
    mask = jnp.arange(MAX_CELLS) <= active[..., None]
    weights = jnp.where(mask, geometry.mobility, 0.0)
    dissolved = total[..., None] * weights / jnp.sum(weights, axis=-1, keepdims=True)
    mm = jnp.where(jnp.arange(MAX_CELLS) == geometry.count[..., None] - 1,
        jnp.asarray(saturation_pct, F64)[..., None] / 100.0 - geometry.upper[..., None],
        geometry.width[..., None]) * volume[..., None]
    return solution_deposit(mm, dissolved, jnp.zeros_like(dissolved), solubility)


def compartment_conductivity(state, theta, thickness, gravel_volume_pct,
                              saturation_pct, field_capacity_pct, solubility=100.0):
    total = jnp.sum(state.dissolved + state.deposited, axis=-1)
    volume = jnp.asarray(thickness, F64) * 1000.0 * (
        1.0 - jnp.asarray(gravel_volume_pct, F64) / 100.0)

    def ec(fraction):
        water = jnp.asarray(fraction, F64) * volume
        concentration = total / jnp.where(water > 0.0, water, 1.0)
        concentration = jnp.where(water > 0.0, concentration,
                                  jnp.where(total > 0.0, solubility, 0.0))
        return jnp.minimum(concentration, solubility) / EQUIV

    return Conductivity(ec(jnp.asarray(saturation_pct, F64) / 100.0), ec(theta),
                         ec(jnp.asarray(field_capacity_pct, F64) / 100.0))


def root_zone_conductivity(conductivity, thickness, rooting_depth, minimum_depth=0.0):
    thickness = jnp.asarray(thickness, F64)
    depth = jnp.asarray(rooting_depth, F64)
    starts = jnp.cumsum(thickness) - thickness
    weights = jnp.clip(depth - starts, 0.0, thickness) / jnp.where(depth > 0.0, depth, 1.0)
    return Conductivity(*[jnp.where((depth >= minimum_depth) & (depth > 0.0),
        jnp.sum(value * weights), -9.0) for value in conductivity])


def salinity_stress(ece, minimum_ece, maximum_ece, shape=0.0, considered=True):
    """global.f90 KsSalinity (linear, convex/concave and shape=99 logistic)."""
    ec, lower, upper, shape = [jnp.asarray(x, F64) for x in
                              (ece, minimum_ece, maximum_ece, shape)]
    relative = (ec - lower) / jnp.where(upper != lower, upper - lower, 1.0)
    rounded = round_fortran(shape * 10.0)
    curved = 1.0 - jnp.expm1(relative * shape) / jnp.expm1(jnp.where(shape != 0.0, shape, 1.0))
    response = jnp.where(rounded == 0, 1.0 - relative,
                         jnp.where(rounded == 990, temperature_stress(upper, lower, ec), curved))
    response = jnp.where(ec <= lower, 1.0, jnp.where(ec >= upper, 0.0, response))
    return jnp.where(considered, jnp.clip(response, 0.0, 1.0), 1.0)


def adjusted_stomatal_stress(ece, ecsw, ecsw_fc, relative_depletion,
                              minimum_ece, maximum_ece, response_pct,
                              coefficients, input_stress, considered=True):
    """global.f90 AdjustedKsStoToECsw, with explicit calibration coefficients."""
    relative = (jnp.asarray(ecsw, F64) - (jnp.asarray(ecsw_fc, F64) - ece)
                + (jnp.asarray(response_pct, F64) - 100.0) * relative_depletion)
    stress = (1.0 - salinity_stress(relative, minimum_ece, maximum_ece, 3.0)) * 100.0
    b0, b1, b2 = coefficients
    adjusted = jnp.maximum(1.0 - (b0 + b1 * stress + b2 * stress * stress) / 100.0, 0.0)
    adjusted = jnp.minimum(adjusted, input_stress)
    result = jnp.where(relative >= maximum_ece, 0.0,
                       jnp.where(relative <= minimum_ece, input_stress, adjusted))
    enabled = considered & (jnp.asarray(response_pct) > 0) & (
        jnp.asarray(relative_depletion) > jnp.finfo(F64).eps)
    return jnp.where(enabled, result, input_stress)


def concentrate_salts(state, theta, thickness, gravel_volume_pct, geometry, solubility=100.0):
    """simul.f90 ConcentrateSalts: ET moves salt into the remaining wet cells."""
    active = active_cells(theta, geometry)
    cells = jnp.arange(MAX_CELLS)
    dry = cells > active[..., None]
    moved = jnp.sum(jnp.where(dry, state.dissolved + state.deposited, 0.0), axis=-1)
    solution = jnp.where(dry, 0.0, state.dissolved)
    deposit = jnp.where(dry, 0.0, state.deposited)
    solution += jnp.where(cells == active[..., None], moved[..., None], 0.0)
    volume = geometry.width * 1000.0 * jnp.asarray(thickness, F64) * (
        1.0 - jnp.asarray(gravel_volume_pct, F64) / 100.0)
    balanced = solution_deposit(volume[..., None], solution, deposit, solubility)
    rebalance = (cells == active[..., None]) & (moved[..., None] > 0.0)
    return SaltState(jnp.where(rebalance, balanced.dissolved, solution),
                     jnp.where(rebalance, balanced.deposited, deposit))
