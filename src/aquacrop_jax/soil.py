"""Soil-profile loading and JAX derivations from global.f90."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp

F64 = jnp.float64


@dataclass(frozen=True)
class SoilLayer:
    thickness: float
    saturation: float
    field_capacity: float
    wilting_point: float
    ksat: float
    penetrability: int
    gravel_mass: int
    capillary_a: float
    capillary_b: float
    description: str

    @property
    def gravel_volume(self) -> float:
        if self.gravel_mass <= 0:
            return 0.0
        mineral_bd = 2.65
        matrix_bd = mineral_bd * (1.0 - self.saturation / 100.0)
        soil_bd = 100.0 / (self.gravel_mass / mineral_bd +
                           (100.0 - self.gravel_mass) / matrix_bd)
        return self.gravel_mass * soil_bd / mineral_bd


@dataclass(frozen=True)
class SoilProfile:
    description: str
    version: float
    curve_number: int
    readily_evaporable_water: int
    layers: tuple[SoilLayer, ...]


@dataclass(frozen=True)
class Compartments:
    thickness: tuple[float, ...]
    layer_index: tuple[int, ...]


def read_soil_profile(path: str | Path) -> SoilProfile:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 9:
        raise ValueError(f"Soil profile is too short: {path}")
    value = lambda n: lines[n].split(":", 1)[0].strip()
    version = float(value(1)); count = int(value(4))
    layers: list[SoilLayer] = []
    for line in lines[8:8 + count]:
        fields = line.split(maxsplit=9)
        if len(fields) < 9:
            raise ValueError(f"Invalid soil layer in {path}: {line!r}")
        if version >= 6.0:
            layers.append(SoilLayer(
                *(float(fields[n]) for n in range(5)),
                int(fields[5]), int(fields[6]), float(fields[7]), float(fields[8]),
                fields[9] if len(fields) == 10 else "",
            ))
        else:
            raise NotImplementedError("progressive port currently supports SOL version >= 6.0")
    return SoilProfile(lines[0].strip(), version, int(value(2)), int(value(3)), tuple(layers))


def discretize_profile(profile: SoilProfile, default_thickness: float = 0.10,
                       maximum: int = 12) -> Compartments:
    """Exact DetermineNrandThicknessCompartments initial discretization."""
    total = sum(layer.thickness for layer in profile.layers)
    thickness: list[float] = []
    while len(thickness) < maximum and abs(sum(thickness) - total) >= 0.0001:
        remaining = total - sum(thickness)
        thickness.append(default_thickness if remaining > default_thickness else remaining)
    layer_index: list[int] = []
    center = 0.0
    boundary = profile.layers[0].thickness
    layer = 0
    for dz in thickness:
        center += dz / 2.0
        while layer + 1 < len(profile.layers) and center > boundary:
            layer += 1; boundary += profile.layers[layer].thickness
        layer_index.append(layer)
        center += dz / 2.0
    return Compartments(tuple(thickness), tuple(layer_index))


def resize_for_root_depth(profile: SoilProfile, compartments: Compartments,
                          root_depth: float, default_thickness: float = 0.10,
                          maximum: int = 12) -> Compartments:
    """Translate the compartment geometry in AdjustSizeCompartments.

    AquaCrop adds 1e-6 m to the crop maximum root depth. Consequently a
    nominal 3.00 m root depth produces a 3.05 m profile after 0.05 m rounding.
    """
    target = root_depth + 0.000001
    dz = list(compartments.thickness)
    total = sum(dz)
    while len(dz) < maximum:
        remaining = target - total
        dz.append(default_thickness if remaining > default_thickness else remaining)
        total += dz[-1]
        if len(dz) == maximum or total + 0.00001 >= target:
            break
    if total + 0.00001 < target:
        factor = (target / 0.1 - 12.0) / 78.0
        dz = [0.05 * int((0.1 * (1.0 + i * factor)) * 20.0 + 0.5)
              for i in range(1, maximum + 1)]
        total = sum(dz)
        while total < target:
            dz[-1] += 0.05; total += 0.05
        while total - 0.04999999 >= target:
            dz[-1] -= 0.05; total -= 0.05
    layer_index: list[int] = []
    center = 0.0; layer = 0; boundary = profile.layers[0].thickness
    for value in dz:
        center += value / 2.0
        while layer + 1 < len(profile.layers) and center > boundary:
            layer += 1; boundary += profile.layers[layer].thickness
        layer_index.append(layer)
        center += value / 2.0
    return Compartments(tuple(dz), tuple(layer_index))


@jax.jit
def drainage_tau(ksat):
    """Exact TauFromKsat with AquaCrop hundredth rounding."""
    ksat = jnp.asarray(ksat, F64)
    raw = 100.0 * 0.0866 * jnp.exp(0.35 * jnp.log(ksat))
    rounded = jnp.floor(raw + 0.5)
    tau = jnp.clip(rounded, 0.0, 100.0) / 100.0
    return jnp.where(jnp.abs(ksat) < jnp.finfo(F64).eps,
                     jnp.zeros_like(tau), tau)


@jax.jit
def profile_water_mm(theta, thickness, gravel_volume_pct):
    theta = jnp.asarray(theta, F64)
    thickness = jnp.asarray(thickness, F64)
    gravel = jnp.asarray(gravel_volume_pct, F64) / 100.0
    return jnp.sum(1000.0 * theta * thickness * (1.0 - gravel))


@jax.jit
def water_balance_residual(begin_profile, begin_surface, end_profile,
                           end_surface, rain, irrigation, capillary_rise,
                           horizontal_inflow, drainage, runoff, evaporation,
                           transpiration):
    """Exact water diagnostic from CheckWaterSaltBalance, in millimetres."""
    values = [jnp.asarray(v, F64) for v in (
        begin_profile, begin_surface, end_profile, end_surface, rain,
        irrigation, capillary_rise, horizontal_inflow, drainage, runoff,
        evaporation, transpiration)]
    (w0, s0, w1, s1, rain, irrigation, rise, horizontal, drainage,
     runoff, evaporation, transpiration) = values
    return (w0 + s0 + rain + irrigation + rise + horizontal -
            w1 - s1 - drainage - runoff - evaporation - transpiration)


@jax.jit
def drainage_delta(theta, adjusted_fc, saturation_pct, field_capacity_pct, tau):
    """Exact scalar/vector translation of simul.f90 calculate_delta_theta."""
    theta = jnp.asarray(theta, F64)
    adjusted_fc = jnp.asarray(adjusted_fc, F64)
    theta_sat = jnp.asarray(saturation_pct, F64) / 100.0
    theta_fc = jnp.asarray(field_capacity_pct, F64) / 100.0
    tau = jnp.asarray(tau, F64)
    limited = jnp.minimum(theta, theta_sat)
    denominator = jnp.expm1(theta_sat - theta_fc)
    delta = tau * (theta_sat - theta_fc) * jnp.expm1(limited - theta_fc) / denominator
    delta = jnp.where(limited <= adjusted_fc / 100.0, jnp.zeros_like(delta), delta)
    delta = jnp.where(limited - delta < adjusted_fc, limited - adjusted_fc, delta)
    return jnp.maximum(delta, jnp.zeros_like(delta))


@jax.jit
def theta_from_drainage(delta_theta, adjusted_fc, saturation_pct,
                        field_capacity_pct, tau):
    """Exact inverse used by simul.f90 calculate_theta."""
    delta = jnp.asarray(delta_theta, F64)
    adjusted_fc = jnp.asarray(adjusted_fc, F64)
    theta_sat = jnp.asarray(saturation_pct, F64) / 100.0
    theta_fc = jnp.asarray(field_capacity_pct, F64) / 100.0
    tau = jnp.asarray(tau, F64)
    eps = jnp.asarray(jnp.finfo(F64).eps, F64)
    positive_tau = tau > 0.0
    safe_tau = jnp.where(positive_tau, tau, jnp.ones_like(tau))
    candidate = theta_fc + jnp.log1p(
        delta * jnp.expm1(theta_sat - theta_fc) /
        (safe_tau * (theta_sat - theta_fc)))
    candidate = jnp.maximum(candidate, adjusted_fc)
    candidate = jnp.where(positive_tau, candidate, theta_sat + 0.1)
    return jnp.where(delta <= eps, adjusted_fc, candidate)


@jax.jit
def vertical_drainage(theta, thickness, gravel_volume_pct, adjusted_fc,
                      saturation_pct, field_capacity_pct, tau, ksat):
    """Propagate drainage through a homogeneous compartment column.

    This is the single-soil-layer path used by Ottawa. Water released by a
    compartment joins the incoming flux; `ksat` limits each outgoing flux.
    The returned bottom flux is profile drainage. Cases requiring saturation
    backflow are rejected until the multihorizon redistribution port lands.
    """
    theta = jnp.asarray(theta, F64)
    thickness = jnp.asarray(thickness, F64)
    gravel = jnp.asarray(gravel_volume_pct, F64) / 100.0
    adjusted_fc = jnp.asarray(adjusted_fc, F64)
    capacity = 1000.0 * thickness * (1.0 - gravel)
    saturation = jnp.asarray(saturation_pct, F64)
    field_capacity = jnp.asarray(field_capacity_pct, F64)
    tau = jnp.asarray(tau, F64)
    ksat = jnp.asarray(ksat, F64)

    def body(incoming, values):
        current, current_capacity, current_fc = values
        delta = drainage_delta(current, current_fc, saturation,
                               field_capacity, tau)
        released = delta * current_capacity
        outgoing = jnp.minimum(incoming + released, ksat)
        updated = (current * current_capacity + incoming - outgoing) / current_capacity
        return outgoing, (updated, outgoing)

    bottom, (output_theta, output_flux) = jax.lax.scan(
        body, jnp.asarray(0.0, F64), (theta, capacity, adjusted_fc))
    return output_theta, output_flux, bottom
