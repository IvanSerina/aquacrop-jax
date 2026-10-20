"""Budget -> actual ET -> production -> next-day fertility, with both balances."""
from ctypes.util import find_library
from pathlib import Path
import shutil
import subprocess
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aquacrop_jax.canopy import (EarlyVegetativeCanopyState, calculate_potential_et,
    early_vegetative_gdd_step, leaf_thresholds_for_eto)
from aquacrop_jax.fertility import soil_fertility_effect
from aquacrop_jax.production import dynamic_fertility_stress_step
from aquacrop_jax.roots import canopy_water_zone
from aquacrop_jax.salinity import SaltState
from aquacrop_jax.water_crop import (WaterCropFeedbackTrajectory,
    WaterCropPostFeedbackTrajectory, run_water_crop_trajectory)

ROOT = Path(__file__).resolve().parents[1]
DAYS = jnp.asarray([70., 80.], jnp.float64)
FORCING = (DAYS, jnp.full(2, 10.), jnp.asarray([6, 7]),
           jnp.full(2, 5.), jnp.asarray([1.03, 1.06]))
THICKNESS = jnp.full(3, .1, jnp.float64)


class CropState(NamedTuple):
    canopy: EarlyVegetativeCanopyState
    biomass: jax.Array
    fertility_stress_pct: jax.Array


class PreDiagnostics(NamedTuple):
    fertility_stress_pct: jax.Array
    canopy: jax.Array
    potential_transpiration: jax.Array
    valid: jax.Array
    water_mm: jax.Array


class PostDiagnostics(NamedTuple):
    biomass: jax.Array
    fertility_stress_pct: jax.Array
    sum_kc_top_stress: jax.Array
    water_mm: jax.Array


def initial_state():
    return CropState(EarlyVegetativeCanopyState(jnp.float64(.08), jnp.float64(.05),
        jnp.float64(.08), jnp.bool_(False)), jnp.float64(.5), jnp.int32(50))


def pre_et(state, water, forcing):
    total, daily, dap, eto, _ = forcing
    # Calibrated crop with no WP reduction isolates the native biomass/stress
    # recurrence while retaining real fertility canopy and actual water ET.
    effect = soil_fertility_effect(state.fertility_stress_pct, 2.35, .79, 25., 6.26)
    zone = canopy_water_zone(water.theta, THICKNESS, jnp.zeros(3), .3, 29., 13., 46., .1)
    upper, lower = leaf_thresholds_for_eto(eto, .15, .55)
    canopy, _, valid = early_vegetative_gdd_step(state.canopy, total, daily,
        5., 590., 1803., 1803., 1803., .05, .95, .012, .006,
        effect.reduction_growth_pct, effect.reduction_cover_pct,
        zone.actual_mm, zone.field_capacity_mm, zone.wilting_point_mm, upper, lower, 3.)
    demand = calculate_potential_et(dap, 1, 39, 164, 164, canopy.cover, eto,
        1.15, 11., .95, jnp.maximum(canopy.cover, state.canopy.cover), 60.,
        369.41, daily, 8., senescence_exponent=1.)
    tpot = jnp.where(valid, demand.transpiration_mm, jnp.nan)
    epot = jnp.where(valid, demand.evaporation_mm, jnp.nan)
    diagnostic = PreDiagnostics(state.fertility_stress_pct, canopy.cover,
        tpot, valid, jnp.sum(water.theta * THICKNESS * 1000.))
    return state._replace(canopy=canopy), tpot, epot, diagnostic


def post_et(state, before, water, forcing):
    _, _, _, eto, unlimited = forcing
    total = state.biomass + .15 * (water.transpiration / eto)
    stress = dynamic_fertility_stress_step(total, unlimited, .5, 0., .4, 0.,
        50, state.fertility_stress_pct, 100.)
    next_state = state._replace(biomass=total, fertility_stress_pct=stress.stress_pct)
    diagnostic = PostDiagnostics(total, stress.stress_pct, stress.sum_kc_top_stress,
                                 jnp.sum(water.theta * THICKNESS * 1000.))
    return next_state, diagnostic


def simulate(theta, rain, forcing, state, *, with_post=True, salt=None,
             surface=0., evaporation_depth=.15, stage2=-1.):
    return run_water_crop_trajectory(theta, rain, forcing[3],
        jnp.full_like(forcing[3], .3), None, THICKNESS, jnp.zeros(3),
        jnp.full(3, .29), 46., 29., 13., 1., 1200., 46, 7., .6, .02, .01,
        stomatal_shape=3., initial_ece=1. if salt is None else None,
        initial_salt=salt, initial_surface_water_mm=surface,
        initial_evap_depth_m=evaporation_depth, initial_stage2_start_pct=stage2,
        daily_feedback_step=pre_et, initial_feedback_state=state,
        feedback_forcing=forcing, daily_post_et_step=post_et if with_post else None)


@pytest.fixture(scope="module")
def coupled():
    args = (jnp.asarray([.14, .16, .16]), jnp.asarray([0., 40.]), FORCING, initial_state())
    return simulate(*args), jax.jit(simulate)(*args)


def test_post_et_carries_native_order_and_uses_actual_fluxes(coupled):
    for result in coupled:
        assert isinstance(result, WaterCropPostFeedbackTrajectory)
        pre, post, water = result.feedback, result.post_feedback, result.water
        np.testing.assert_array_equal(pre.valid, True)
        assert int(pre.fertility_stress_pct[0]) == 50
        np.testing.assert_array_equal(pre.fertility_stress_pct[1:], post.fertility_stress_pct[:-1])
        assert int(post.fertility_stress_pct[0]) != 50
        assert float(water.transpiration[0]) < float(pre.potential_transpiration[0])
        np.testing.assert_allclose(post.biomass,
            .5 + .15 * np.cumsum(np.asarray(water.transpiration) / 5.), rtol=0., atol=1e-14)
        np.testing.assert_allclose(pre.water_mm - post.water_mm,
            water.evaporation + water.transpiration, rtol=0., atol=1e-11)
        np.testing.assert_allclose(water.residual, 0., rtol=0., atol=1e-9)
        np.testing.assert_allclose(water.salt_residual, 0., rtol=0., atol=1e-9)
        assert post.biomass.dtype == jnp.float64
        assert post.fertility_stress_pct.dtype == jnp.int32
        np.testing.assert_allclose(result.final_feedback_state.biomass, post.biomass[-1], rtol=0., atol=1e-14)


def test_post_et_stress_matches_original_native_production(coupled, tmp_path):
    reference = ROOT.parent / "aquacrop-fortran/src"
    objects = sorted(p for p in reference.glob("*.o") if p.name != "aquacrop.o")
    if not shutil.which("gfortran") or not find_library("gfortran") or not objects:
        pytest.skip("Native production probe requires gfortran and reference objects")
    probe = tmp_path / "probe"
    compiled = subprocess.run(["gfortran", "-O0", "-I", str(reference), "-o", str(probe),
        str(ROOT / "scripts/fertility_stress_probe.f90"), *map(str, objects)], capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    result = coupled[1]
    previous, total, native = 50, .5, []
    for day in range(2):
        row = [50, 1, total, float(FORCING[4][day]), .5, 0., .4, 0., previous, 100.,
               5., float(result.water.transpiration[day]), 0, 0, 1, 0, 100, 100, 0]
        captured = subprocess.run([str(probe)], input="1\n" + " ".join(
            format(value, ".17g") for value in row) + "\n", capture_output=True, text=True, check=True)
        values = list(map(float, captured.stdout.split()))
        previous, total = int(values[1]), values[2]
        native.append(values)
    native = np.asarray(native)
    np.testing.assert_array_equal(result.post_feedback.fertility_stress_pct, native[:, 0])
    np.testing.assert_allclose(result.post_feedback.biomass, native[:, 2], rtol=0., atol=1e-14)
    np.testing.assert_allclose(result.post_feedback.sum_kc_top_stress, native[:, 4], rtol=0., atol=1e-14)


def test_pre_only_contract_and_next_day_canopy_feedback(coupled):
    pre_only = jax.jit(lambda: simulate(jnp.asarray([.14, .16, .16]),
        jnp.asarray([0., 40.]), FORCING, initial_state(), with_post=False))()
    assert isinstance(pre_only, WaterCropFeedbackTrajectory)
    assert len(pre_only) == 2
    full = coupled[1]
    np.testing.assert_allclose(full.water.theta[0], pre_only.water.theta[0], rtol=0., atol=1e-14)
    assert float(full.feedback.canopy[1]) > float(pre_only.feedback.canopy[1])


def test_post_et_state_restarts_with_soil_and_salt(coupled):
    operation = jax.jit(simulate)
    first = operation(jnp.asarray([.14, .16, .16]), jnp.asarray([0.]),
        tuple(value[:1] for value in FORCING), initial_state())
    water = first.water
    second = operation(water.theta[-1], jnp.asarray([40.]),
        tuple(value[1:] for value in FORCING), first.final_feedback_state,
        salt=SaltState(water.salt_dissolved[-1], water.salt_deposited[-1]),
        surface=water.evap_surface_water[-1], evaporation_depth=water.evap_depth[-1],
        stage2=water.stage2_start[-1])
    full = coupled[1]
    np.testing.assert_allclose(second.water.theta[-1], full.water.theta[-1], rtol=0., atol=1e-12)
    np.testing.assert_allclose(second.water.salt_dissolved[-1], full.water.salt_dissolved[-1], rtol=0., atol=1e-12)
    np.testing.assert_allclose(second.final_feedback_state.biomass,
                               full.final_feedback_state.biomass, rtol=0., atol=1e-14)
    np.testing.assert_array_equal(second.final_feedback_state.fertility_stress_pct,
                                  full.final_feedback_state.fertility_stress_pct)


def test_post_et_requires_pre_et_state_contract():
    with pytest.raises(ValueError, match="post-ET feedback requires"):
        run_water_crop_trajectory([.2], [0.], [0.], [0.], [0.], [.1], [0.], [.29],
            46., 29., 13., 1., 1200., 46, 7., .6, .02, .01, daily_post_et_step=post_et)
