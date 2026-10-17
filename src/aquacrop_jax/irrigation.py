"""Pure JAX irrigation scheduling operators translated from AquaCrop."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

F64 = jnp.float64


class GeneratedIrrigation(NamedTuple):
    depth_mm: jax.Array
    triggered: jax.Array
    projected_root_water_mm: jax.Array
    trigger_depletion_mm: jax.Array


class NetIrrigationResult(NamedTuple):
    theta: jax.Array
    depth_mm: jax.Array
    threshold_mm: jax.Array


def manual_irrigation(day_number, event_days, event_depths_mm):
    """Return the sum of manual events matching an absolute day number."""
    days = jnp.asarray(event_days, jnp.int32)
    depths = jnp.asarray(event_depths_mm, F64)
    if days.shape != depths.shape:
        raise ValueError("event day and depth arrays must have equal shape")
    return jnp.sum(jnp.where(days == jnp.asarray(day_number, jnp.int32),
                             jnp.maximum(depths, 0.0), 0.0))


def generated_irrigation(root_actual_mm, root_fc_mm, root_threshold_mm,
                         target_time_value, target_depth_mm,
                         mode="raw", depth_mode="fixed",
                         potential_evaporation_mm=0.0,
                         potential_transpiration_mm=0.0, rain_mm=0.0,
                         runoff_mm=0.0, subdrain_mm=0.0):
    """Translate ``Calculate_irrigation`` without global state.

    ``mode='depletion'`` interprets ``target_time_value`` as millimetres.
    ``mode='raw'`` interprets it as a percentage of readily available water.
    ``depth_mode='fixed'`` applies ``target_depth_mm``; ``refill`` fills the
    projected root zone to field capacity plus the configured offset.
    """
    actual, fc, threshold = [jnp.asarray(v, F64) for v in
                             (root_actual_mm, root_fc_mm, root_threshold_mm)]
    projected = (actual - jnp.asarray(potential_evaporation_mm, F64)
                 - jnp.asarray(potential_transpiration_mm, F64)
                 + jnp.asarray(rain_mm, F64) - jnp.asarray(runoff_mm, F64)
                 - jnp.asarray(subdrain_mm, F64))
    target = jnp.asarray(target_time_value, F64)
    if mode == "depletion":
        trigger_depletion = target
    elif mode == "raw":
        trigger_depletion = target / 100.0 * (fc - threshold)
    else:
        raise ValueError("mode must be 'depletion' or 'raw'")
    triggered = fc - projected >= trigger_depletion
    if depth_mode == "fixed":
        depth = jnp.asarray(target_depth_mm, F64)
    elif depth_mode == "refill":
        depth = fc - projected + jnp.asarray(target_depth_mm, F64)
    else:
        raise ValueError("depth_mode must be 'fixed' or 'refill'")
    depth = jnp.where(triggered, jnp.maximum(depth, 0.0), 0.0)
    return GeneratedIrrigation(depth, triggered, projected, trigger_depletion)


def net_irrigation(theta, thickness, gravel_volume_pct, field_capacity_pct,
                   critical_theta, rooting_depth, allowable_raw_pct):
    """Raise rooted compartments to the Inet threshold conservatively."""
    theta = jnp.asarray(theta, F64)
    thickness = jnp.asarray(thickness, F64)
    gravel = jnp.asarray(gravel_volume_pct, F64) / 100.0
    fc = jnp.asarray(field_capacity_pct, F64) / 100.0
    critical = jnp.asarray(critical_theta, F64)
    root_depth = jnp.maximum(jnp.asarray(rooting_depth, F64), 0.0)
    starts = jnp.concatenate((jnp.zeros((1,), F64), jnp.cumsum(thickness)[:-1]))
    fractions = jnp.clip((root_depth - starts) / thickness, 0.0, 1.0)
    threshold = fc - jnp.asarray(allowable_raw_pct, F64) / 100.0 * (fc - critical)
    delta_theta = jnp.maximum(threshold - theta, 0.0) * fractions
    capacity = 1000.0 * thickness * (1.0 - gravel)
    depth = jnp.sum(delta_theta * capacity)
    return NetIrrigationResult(theta + delta_theta, depth, threshold)
