"""Coupled JAX daily water, evaporation and root-transpiration recurrence."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .evaporation import stage1_evaporation, stage2_evaporation
from .capillary import capillary_rise
from .groundwater import (adjusted_field_capacity, constrain_root_depth,
                          saturate_below_water_table)
from .hydrology import daily_water_step_irrigated
from .soil import profile_water_mm, water_balance_residual
from .transpiration import compartment_transpiration
from .water_evap import prepare_stage2_percentage
from .salinity import (EQUIV, SaltState, active_cells, salt_geometry,
    initialize_salt, compartment_conductivity, concentrate_salts,
    root_zone_conductivity, salinity_stress, round_fortran)
from .salt_transport import salt_transport_step

F64 = jnp.float64


class WaterCropTrajectory(NamedTuple):
    theta: jax.Array
    runoff: jax.Array
    drainage: jax.Array
    infiltration: jax.Array
    evaporation: jax.Array
    transpiration: jax.Array
    transpiration_stress: jax.Array
    evap_surface_water: jax.Array
    evap_depth: jax.Array
    stage2_start: jax.Array
    residual: jax.Array
    irrigation: jax.Array
    capillary_rise: jax.Array
    groundwater_inflow: jax.Array
    groundwater_depth: jax.Array
    salt_dissolved: jax.Array
    salt_deposited: jax.Array
    salt_infiltrated: jax.Array
    salt_drained: jax.Array
    salt_groundwater: jax.Array
    salt_residual: jax.Array
    ece: jax.Array
    rootzone_ece: jax.Array
    salinity_stress: jax.Array
    mean_salinity_stress_pct: jax.Array
    salinity_growing_days: jax.Array
    salinity_stomatal_reduction_pct: jax.Array


class WaterCropPreET(NamedTuple):
    """Daily state after infiltration, capillary rise and salt transport."""

    theta: jax.Array
    salt: SaltState
    water: object
    begin_water_mm: jax.Array
    begin_salt_g_m2: jax.Array
    groundwater_inflow_mm: jax.Array
    capillary_rise_mm: jax.Array
    salt_infiltrated_g_m2: jax.Array
    salt_drained_g_m2: jax.Array
    salt_groundwater_g_m2: jax.Array


def water_crop_pre_et_step(
        current_theta, salt, rain_day, irrigation_day, groundwater_day,
        groundwater_ec_day, irrigation_ec_day, *, thickness, gravel_volume_pct,
        saturation_pct, field_capacity_pct, wilting_point_pct, adjusted_fc,
        tau, ksat, curve_number, initial_abstraction_pct, capillary_enabled,
        cap_a, cap_b, salt_enabled, geometry, salt_solubility):
    """Apply the water and salt part of one day, before canopy and ET.

    Inputs and outputs are JAX pytrees with static compartment/cell shapes.
    The returned soil and salt state is the one used by the Fortran canopy
    step, before evaporation and root extraction modify it.
    """
    salt_begin = jnp.sum(salt.dissolved + salt.deposited)
    begin = profile_water_mm(current_theta, thickness, gravel_volume_pct)
    saturated_theta = saturate_below_water_table(
        current_theta, thickness, saturation_pct, groundwater_day)
    groundwater_inflow = profile_water_mm(
        saturated_theta, thickness, gravel_volume_pct) - begin
    daily_adjusted_fc = adjusted_field_capacity(
        thickness, field_capacity_pct, saturation_pct,
        groundwater_day) / 100.0
    water = daily_water_step_irrigated(
        saturated_theta, thickness, gravel_volume_pct,
        jnp.where(groundwater_day >= 0.0, daily_adjusted_fc, adjusted_fc),
        saturation_pct, field_capacity_pct, tau, ksat, rain_day,
        irrigation_day,
        curve_number, initial_abstraction_pct)
    if capillary_enabled:
        capillary = capillary_rise(
            water.theta, thickness, gravel_volume_pct,
            jnp.where(groundwater_day >= 0.0,
                      daily_adjusted_fc * 100.0,
                      jnp.asarray(adjusted_fc, F64) * 100.0),
            field_capacity_pct, wilting_point_pct, ksat, cap_a, cap_b,
            groundwater_day)
        water_theta = capillary.theta
        capillary_mm = capillary.water_mm
    else:
        water_theta = water.theta
        capillary_mm = jnp.asarray(0.0, F64)
    salt_in = salt_out = salt_gw = jnp.asarray(0.0, F64)
    if salt_enabled:
        added_water = (water_theta - water.theta) * (1000.0 * jnp.asarray(thickness, F64) *
            (1.0 - jnp.asarray(gravel_volume_pct, F64) / 100.0))
        added_water += (saturated_theta - current_theta) * (1000.0 * jnp.asarray(thickness, F64) *
            (1.0 - jnp.asarray(gravel_volume_pct, F64) / 100.0))
        salt_added = added_water * groundwater_ec_day * EQUIV
        active = active_cells(water_theta, geometry)
        salt = SaltState(salt.dissolved + jnp.where(
            jnp.arange(11) == active[:, None], salt_added[:, None], 0.0), salt.deposited)
        salt_gw = jnp.sum(salt_added)
        # Legacy extra-runoff logic gives irrigation priority over rainfall.
        scs_infiltration = jnp.maximum(rain_day + irrigation_day - water.runoff, 0.0)
        accepted_irrigation = jnp.minimum(irrigation_day, scs_infiltration)
        salt_step = salt_transport_step(salt, water_theta, water.fluxout,
            thickness, gravel_volume_pct, saturation_pct, field_capacity_pct, geometry,
            infiltrated_rain=scs_infiltration - accepted_irrigation,
            infiltrated_irrigation=accepted_irrigation,
            irrigation_ec=irrigation_ec_day, solubility=salt_solubility)
        salt = salt_step.state
        salt_in, salt_out = salt_step.infiltrated_g_m2, salt_step.drained_g_m2
    return WaterCropPreET(water_theta, salt, water, begin, salt_begin,
                          groundwater_inflow, capillary_mm, salt_in, salt_out,
                          salt_gw)


def run_water_crop_trajectory(
        initial_theta, rain, eto, rooting_depth, potential_transpiration,
        thickness, gravel_volume_pct, adjusted_fc, saturation_pct,
        field_capacity_pct, wilting_point_pct, tau, ksat, curve_number,
        rew_mm, p_stomatal, smax_top, smax_bottom, stomatal_shape=0.0,
        stress_reduction_pct=0.0, sink_correction=1.0,
        bare_soil_coefficient=1.10, initial_abstraction_pct=5.0,
        maximum_evap_depth_m=0.30, decline_factor=4.0,
        initial_surface_water_mm=0.0, initial_evap_depth_m=0.15,
        initial_stage2_start_pct=-1.0, daily_irrigation=None,
        groundwater_depth_m=None, capillary_a=None, capillary_b=None,
        minimum_root_depth_m=0.0, anaerobic_sensitive=True,
        initial_salt=None, initial_ece=None, irrigation_ec=None,
        groundwater_ec=None, salt_diffusion_pct=20.0, salt_solubility=100.0,
        salinity_minimum_ece=2.0, salinity_maximum_ece=16.0,
        salinity_response_pct=100.0, salt_stomatal_coefficients=None,
        potential_evaporation=None, salinity_growing_mask=None,
        initial_salinity_growing_days=0.0, initial_mean_salinity_stress_pct=0.0):
    """Run explicit daily water/salt state with optional calibrated salt stress.

    ``potential_evaporation`` enables canopy-shaded demand. Salt stores and
    growing-day stress averages can be resumed using the returned histories.
    Primary ECe and additional ECsw stomatal responses require the reference
    calibration polynomial; no coefficients are inferred here.
    """
    initial_theta = jnp.asarray(initial_theta, F64)
    rain = jnp.asarray(rain, F64)
    eto = jnp.asarray(eto, F64)
    rooting_depth = jnp.asarray(rooting_depth, F64)
    potential_transpiration = jnp.asarray(potential_transpiration, F64)
    adjusted_fc = jnp.asarray(adjusted_fc, F64)
    irrigation = (jnp.zeros_like(rain) if daily_irrigation is None else
                  jnp.asarray(daily_irrigation, F64))
    groundwater = (jnp.full_like(rain, -1.0) if groundwater_depth_m is None else
                   jnp.asarray(groundwater_depth_m, F64))
    if not (rain.shape == eto.shape == rooting_depth.shape ==
            potential_transpiration.shape == irrigation.shape ==
            groundwater.shape):
        raise ValueError("daily forcing arrays must have equal shape")
    capillary_enabled = capillary_a is not None and capillary_b is not None
    cap_a = (jnp.zeros_like(initial_theta) if capillary_a is None else
             jnp.asarray(capillary_a, F64))
    cap_b = (jnp.zeros_like(initial_theta) if capillary_b is None else
             jnp.asarray(capillary_b, F64))
    salt_enabled = any(x is not None for x in
                       (initial_salt, initial_ece, irrigation_ec, groundwater_ec))
    geometry = salt_geometry(jnp.broadcast_to(jnp.asarray(saturation_pct, F64), initial_theta.shape),
        jnp.broadcast_to(jnp.asarray(field_capacity_pct, F64), initial_theta.shape),
        jnp.broadcast_to(jnp.asarray(ksat, F64), initial_theta.shape), salt_diffusion_pct)
    if initial_salt is None:
        salt_initial = initialize_salt(
            0.0 if initial_ece is None else initial_ece, initial_theta,
            thickness, gravel_volume_pct, saturation_pct, geometry, salt_solubility)
    else:
        salt_initial = SaltState(*[jnp.asarray(x, F64) for x in initial_salt])
        if any(x.shape != (initial_theta.size, 11) for x in salt_initial):
            raise ValueError("salt stores must have shape (compartments, 11)")
    if initial_salt is not None and initial_ece is not None:
        raise ValueError("Provide initial_salt or initial_ece, not both")
    growing = (rooting_depth > 0.0 if salinity_growing_mask is None else
               jnp.asarray(salinity_growing_mask, jnp.bool_))
    if growing.shape != rain.shape:
        raise ValueError("salinity growing mask must match daily forcing")
    irri_ec = jnp.zeros_like(rain) if irrigation_ec is None else jnp.asarray(irrigation_ec, F64)
    gw_ec = jnp.zeros_like(rain) if groundwater_ec is None else jnp.asarray(groundwater_ec, F64)
    epot = (eto * jnp.asarray(bare_soil_coefficient, F64) if potential_evaporation is None
            else jnp.asarray(potential_evaporation, F64))
    if not (irri_ec.shape == gw_ec.shape == epot.shape == rain.shape):
        raise ValueError("daily EC and evaporation arrays must match rain shape")
    initial_depth = jnp.asarray(initial_evap_depth_m, F64)
    calculated = prepare_stage2_percentage(
        initial_theta, thickness, gravel_volume_pct, saturation_pct,
        field_capacity_pct, rew_mm, initial_depth)
    supplied = jnp.asarray(initial_stage2_start_pct, F64)
    initial_start = jnp.where(supplied < 0.0, calculated, supplied)

    def body(carry, forcing):
        (current_theta, surface_store, evaporation_depth, stage2_start, salt,
         salt_growing_days, mean_salt_stress) = carry
        (rain_day, eto_day, root_day, tpot_day, irrigation_day,
         groundwater_day, irrigation_ec_day, groundwater_ec_day, epot_day, growing_day) = forcing
        pre_et = water_crop_pre_et_step(
            current_theta, salt, rain_day, irrigation_day, groundwater_day,
            groundwater_ec_day, irrigation_ec_day, thickness=thickness,
            gravel_volume_pct=gravel_volume_pct, saturation_pct=saturation_pct,
            field_capacity_pct=field_capacity_pct,
            wilting_point_pct=wilting_point_pct, adjusted_fc=adjusted_fc,
            tau=tau, ksat=ksat, curve_number=curve_number,
            initial_abstraction_pct=initial_abstraction_pct,
            capillary_enabled=capillary_enabled, cap_a=cap_a, cap_b=cap_b,
            salt_enabled=salt_enabled, geometry=geometry,
            salt_solubility=salt_solubility)
        (water_theta, salt, water, begin, salt_begin, groundwater_inflow,
         capillary_mm, salt_in, salt_out, salt_gw) = pre_et
        salt_reduction = jnp.asarray(0.0, F64)
        if salt_stomatal_coefficients is not None:
            before_et_ec = compartment_conductivity(salt, water_theta, thickness,
                gravel_volume_pct, saturation_pct, field_capacity_pct, salt_solubility)
            before_et_zone = root_zone_conductivity(before_et_ec, thickness, root_day,
                                                     minimum_root_depth_m)
            instantaneous = (1.0 - salinity_stress(before_et_zone.ece,
                salinity_minimum_ece, salinity_maximum_ece)) * 100.0
            mean_salt_stress = jnp.where(growing_day,
                (salt_growing_days * mean_salt_stress + instantaneous) / (salt_growing_days + 1.0),
                mean_salt_stress)
            salt_growing_days += growing_day.astype(F64)
            b0, b1, b2 = salt_stomatal_coefficients
            salt_reduction = round_fortran(b0 + b1 * mean_salt_stress + b2 * mean_salt_stress ** 2)
            salt_reduction = jnp.where((salt_reduction < 0.0) | (mean_salt_stress <= 0.1), 0.0,
                jnp.where((salt_reduction > 100.0) | (mean_salt_stress >= 99.9), 100.0, salt_reduction))
            salt_reduction = jnp.where(growing_day, salt_reduction, 0.0)
        rew = jnp.asarray(rew_mm, F64)
        is_wet = rain_day + irrigation_day > 0.0
        surface_store = jnp.where(
            is_wet, jnp.minimum(jnp.maximum(
                rain_day + irrigation_day - water.runoff, 0.0), rew),
            surface_store)
        evaporation_depth = jnp.where(is_wet, jnp.asarray(0.15, F64),
                                      evaporation_depth)
        first = stage1_evaporation(
            water_theta, thickness, gravel_volume_pct, wilting_point_pct,
            epot_day, surface_store,
            0.15)
        recalculated = prepare_stage2_percentage(
            first.theta, thickness, gravel_volume_pct, saturation_pct,
            field_capacity_pct, rew, 0.15)
        stage2_start = jnp.where(
            jnp.logical_and(is_wet, first.remaining_surface_water < 1e-12),
            recalculated, stage2_start)
        second = stage2_evaporation(
            first.theta, thickness, gravel_volume_pct, saturation_pct,
            field_capacity_pct, wilting_point_pct, rew, stage2_start,
            first.remaining_potential, evaporation_depth,
            maximum_evap_depth_m, decline_factor)
        evaporation = first.actual_evaporation + second.actual_evaporation
        effective_root = constrain_root_depth(
            root_day, groundwater_day, minimum_root_depth_m,
            anaerobic_sensitive)
        salt_ec = compartment_conductivity(salt, second.theta, thickness,
            gravel_volume_pct, saturation_pct, field_capacity_pct, salt_solubility)
        crop = compartment_transpiration(
            second.theta, thickness, gravel_volume_pct, field_capacity_pct,
            wilting_point_pct, saturation_pct, effective_root, tpot_day,
            p_stomatal, smax_top, smax_bottom, stomatal_shape,
            jnp.maximum(jnp.asarray(stress_reduction_pct, F64), salt_reduction), sink_correction,
            salt_conductivity=salt_ec if salt_stomatal_coefficients is not None else None,
            salinity_minimum_ece=salinity_minimum_ece,
            salinity_maximum_ece=salinity_maximum_ece,
            salinity_response_pct=salinity_response_pct,
            salt_stomatal_coefficients=salt_stomatal_coefficients)
        end = profile_water_mm(crop.theta, thickness, gravel_volume_pct)
        residual = water_balance_residual(
            begin, 0.0, end, 0.0, rain_day, irrigation_day, capillary_mm,
            groundwater_inflow,
            water.drainage, water.runoff, evaporation, crop.actual_mm)
        if salt_enabled:
            salt = concentrate_salts(salt, crop.theta, thickness, gravel_volume_pct,
                                     geometry, salt_solubility)
        salt_ec = compartment_conductivity(salt, crop.theta, thickness,
            gravel_volume_pct, saturation_pct, field_capacity_pct, salt_solubility)
        zone_ec = root_zone_conductivity(salt_ec, thickness, effective_root,
                                         minimum_root_depth_m)
        salt_residual = salt_begin + salt_in + salt_gw - salt_out - jnp.sum(salt.dissolved + salt.deposited)
        output = (crop.theta, water.runoff, water.drainage,
                  water.infiltration, evaporation, crop.actual_mm,
                  crop.root_zone_stress, first.remaining_surface_water,
                  second.evaporation_depth, stage2_start, residual,
                  irrigation_day, capillary_mm, groundwater_inflow,
                  groundwater_day, salt.dissolved, salt.deposited, salt_in,
                  salt_out, salt_gw, salt_residual, salt_ec.ece, zone_ec.ece,
                  salinity_stress(zone_ec.ece, salinity_minimum_ece, salinity_maximum_ece),
                  mean_salt_stress, salt_growing_days, salt_reduction)
        carry = (crop.theta, first.remaining_surface_water,
                 second.evaporation_depth, stage2_start, salt, salt_growing_days, mean_salt_stress)
        return carry, output

    _, output = lax.scan(
        body,
        (initial_theta, jnp.asarray(initial_surface_water_mm, F64),
         initial_depth, initial_start, salt_initial,
         jnp.asarray(initial_salinity_growing_days, F64),
         jnp.asarray(initial_mean_salinity_stress_pct, F64)),
        (rain, eto, rooting_depth, potential_transpiration, irrigation,
         groundwater, irri_ec, gw_ec, epot, growing))
    return WaterCropTrajectory(*output)
