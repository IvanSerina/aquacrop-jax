"""Pre-ET actual canopy -> actual production -> cutting -> next-day feedback."""
from ctypes.util import find_library
from pathlib import Path
import shutil
import subprocess
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aquacrop_jax.canopy import (EarlyVegetativeCanopyState, WaterSenescenceState,
                                calculate_potential_et)
from aquacrop_jax.canopy_step import (CanopyDayState, CanopyDayParameters,
                                     active_canopy_gdd_step)
from aquacrop_jax.cuttings import CuttingState, apply_cutting_event, manual_cutting_schedule_step
from aquacrop_jax.fertility import FertilityEffect
from aquacrop_jax.production import fertility_water_productivity_step
from aquacrop_jax.roots import canopy_water_zone
from aquacrop_jax.salinity import SaltState
from aquacrop_jax.water_crop import run_water_crop_trajectory

ROOT = Path(__file__).resolve().parents[1]
THICKNESS = jnp.full(3, .1, jnp.float64)
FORCING = (jnp.asarray([650., 660., 670.]), jnp.full(3, 10.),
           jnp.asarray([70., 71., 72.]), jnp.asarray([100, 101, 102]), jnp.full(3, 5.))


class FeedbackState(NamedTuple):
    canopy: CanopyDayState
    cutting: CuttingState
    next_cut_index: jax.Array
    biomass: jax.Array
    sum_kci: jax.Array


class PreDiagnostics(NamedTuple):
    cover: jax.Array
    potential_transpiration: jax.Array
    valid: jax.Array


class PostDiagnostics(NamedTuple):
    biomass: jax.Array
    water_productivity: jax.Array
    sum_kci: jax.Array
    next_cover: jax.Array
    harvest: object


def initial_state():
    canopy = CanopyDayState(EarlyVegetativeCanopyState(
        jnp.float64(.7), jnp.float64(.05), jnp.float64(.7), jnp.bool_(False)),
        WaterSenescenceState(jnp.float64(0.), jnp.float64(0.), jnp.float64(.7),
            jnp.float64(0.), jnp.bool_(False), jnp.bool_(False)))
    cuts = CuttingState(jnp.int32(0), jnp.int32(-9), jnp.int32(10),
        jnp.float64(100.), jnp.float64(0.), jnp.float64(0.), jnp.float64(.7))
    return FeedbackState(canopy, cuts, jnp.int32(0), jnp.float64(.5), jnp.float64(0.))


def parameters():
    return CanopyDayParameters((1, 60, 180, 210), (5, 600, 1920, 2200),
        1920., .05, .05, .95, .12, .012, .06, .006, .15, .55, 3.,
        .65, 3., 40., True, 1., 10.)


def pre_et(state, water, forcing):
    cumulative, daily, calendar, _, eto = forcing
    zone = canopy_water_zone(water.theta, THICKNESS, jnp.zeros(3), .3,
                             29., 13., 46., .1)
    # Controlled response reduces WP alone, isolating the post-production cut.
    effect = FertilityEffect(jnp.float64(0.), jnp.float64(0.), jnp.float64(50.),
                            jnp.float64(0.), jnp.float64(0.))
    canopy, diagnostic = active_canopy_gdd_step(state.canopy, cumulative, daily,
        calendar, eto, zone.actual_mm, zone.field_capacity_mm, zone.wilting_point_mm,
        parameters(), effect)
    demand = calculate_potential_et(calendar, 1, 60, 180, 210, canopy.canopy.cover,
        eto, 1.15, 11., .95, canopy.senescence.withered_cover,
        60., 369.41, daily, 8., senescence_exponent=1.)
    tpot = jnp.where(diagnostic.valid, demand.transpiration_mm, jnp.nan)
    epot = jnp.where(diagnostic.valid, demand.evaporation_mm, jnp.nan)
    return state._replace(canopy=canopy), tpot, epot, PreDiagnostics(
        canopy.canopy.cover, tpot, diagnostic.valid)


def simulate(theta, forcing, state, scheduled_days, *, salt=None, surface=0.,
             evaporation_depth=.15, stage2=-1.):
    def post_et(state, before, water, row):
        _, daily, calendar, day, eto = row
        production = fertility_water_productivity_step(.15, state.sum_kci, 10., 50.,
                                                        water.transpiration, eto)
        biomass = state.biomass + production.biomass_increment
        cuts = state.cutting._replace(interval_days=state.cutting.interval_days + 1,
            interval_gdd=state.cutting.interval_gdd + daily)
        decision = manual_cutting_schedule_step(state.next_cut_index, day,
            scheduled_days, crop_first_day=31)
        # This controlled forage has HI=100, hence cumulative yield=biomass.
        canopy, cuts, harvest = apply_cutting_event(state.canopy, cuts, decision.occurred,
            calendar, 25, biomass, biomass)
        next_state = FeedbackState(canopy, cuts, decision.next_index,
                                   biomass, production.sum_kci)
        return next_state, PostDiagnostics(biomass, production.water_productivity,
            production.sum_kci, canopy.canopy.cover, harvest)

    return run_water_crop_trajectory(theta, jnp.zeros_like(forcing[4]), forcing[4],
        jnp.full_like(forcing[4], .3), None, THICKNESS, jnp.zeros(3),
        jnp.full(3, .29), 46., 29., 13., 1., 1200., 46, 7., .6, .02, .01,
        stomatal_shape=3., initial_ece=1. if salt is None else None, initial_salt=salt,
        initial_surface_water_mm=surface, initial_evap_depth_m=evaporation_depth,
        initial_stage2_start_pct=stage2, daily_feedback_step=pre_et,
        initial_feedback_state=state, feedback_forcing=forcing, daily_post_et_step=post_et)


@pytest.fixture(scope='module')
def coupled():
    operation = jax.jit(simulate)
    args = (jnp.full(3, .29), FORCING, initial_state())
    return operation(*args, jnp.asarray([100], jnp.int32)), operation(*args, jnp.empty(0, jnp.int32))


def test_cutting_changes_next_day_demand_after_actual_production(coupled):
    cut, uncut = coupled
    np.testing.assert_array_equal(cut.feedback.valid, True)
    np.testing.assert_array_equal(cut.post_feedback.harvest.occurred, [True, False, False])
    np.testing.assert_allclose(cut.water.theta[0], uncut.water.theta[0], rtol=0., atol=1e-14)
    assert float(cut.feedback.cover[0]) > .25
    assert float(cut.post_feedback.next_cover[0]) == .25
    assert float(cut.feedback.cover[1]) < float(uncut.feedback.cover[1])
    assert float(cut.water.transpiration[1]) < float(uncut.water.transpiration[1])
    assert float(cut.post_feedback.harvest.biomass_harvested[0]) > .5
    np.testing.assert_allclose(cut.post_feedback.harvest.biomass_harvested[0],
        cut.post_feedback.biomass[0], rtol=0., atol=1e-14)
    assert int(cut.post_feedback.harvest.interval_days[0]) == 11
    # Cutting resets its own counters; native SumKci continues accumulating.
    np.testing.assert_allclose(cut.post_feedback.sum_kci,
        np.cumsum(np.asarray(cut.water.transpiration) / 5.), rtol=0., atol=1e-14)
    for result in coupled:
        np.testing.assert_allclose(result.water.residual, 0., rtol=0., atol=1e-9)
        np.testing.assert_allclose(result.water.salt_residual, 0., rtol=0., atol=1e-9)


def test_cut_feedback_production_matches_original_actual_fluxes(coupled, tmp_path):
    reference = ROOT.parent / 'aquacrop-fortran/src'
    objects = sorted(p for p in reference.glob('*.o') if p.name != 'aquacrop.o')
    if not shutil.which('gfortran') or not find_library('gfortran') or not objects:
        pytest.skip('Requires original native objects and gfortran')
    probe = tmp_path / 'probe'
    built = subprocess.run(['gfortran', '-O0', '-I', str(reference), '-o', str(probe),
        str(ROOT / 'scripts/fertility_stress_probe.f90'), *map(str, objects)], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    result = coupled[0]
    cumulative, expected = 0., []
    for tr in np.asarray(result.water.transpiration):
        inputs = [50, 1, .5, 1., .5, 0., .4, 0., 50, 100., 5., tr, 0, 0, 1, 0, 100, 100, 0]
        text = '1\n' + ' '.join(format(v, '.17g') for v in inputs) + f'\n50 {cumulative:.17g} 10\n'
        native = subprocess.run([str(probe), '--fertility-wp'], input=text,
            capture_output=True, text=True, check=True)
        values = list(map(float, native.stdout.split()))
        cumulative = values[1]
        expected.append(values)
    expected = np.asarray(expected)
    np.testing.assert_allclose(result.post_feedback.water_productivity, expected[:, 0], rtol=0., atol=1e-14)
    np.testing.assert_allclose(result.post_feedback.sum_kci, expected[:, 1], rtol=0., atol=1e-14)
    np.testing.assert_allclose(result.post_feedback.biomass, .5 + np.cumsum(expected[:, 2]), rtol=0., atol=1e-14)


def test_water_salt_and_crop_state_restart_after_cut(coupled):
    operation = jax.jit(simulate)
    schedule = jnp.asarray([100], jnp.int32)
    first = operation(jnp.full(3, .29), tuple(v[:1] for v in FORCING), initial_state(), schedule)
    water = first.water
    second = operation(water.theta[-1], tuple(v[1:] for v in FORCING), first.final_feedback_state,
        schedule, salt=SaltState(water.salt_dissolved[-1], water.salt_deposited[-1]),
        surface=water.evap_surface_water[-1], evaporation_depth=water.evap_depth[-1], stage2=water.stage2_start[-1])
    whole = coupled[0]
    np.testing.assert_allclose(second.water.theta[-1], whole.water.theta[-1], rtol=0., atol=1e-12)
    np.testing.assert_allclose(second.water.salt_dissolved[-1], whole.water.salt_dissolved[-1], rtol=0., atol=1e-12)
    for a, b in zip(jax.tree.leaves(second.final_feedback_state), jax.tree.leaves(whole.final_feedback_state)):
        np.testing.assert_allclose(a, b, rtol=0., atol=1e-14)
