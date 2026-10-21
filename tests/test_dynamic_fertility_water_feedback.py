"""Controlled actual SF -> canopy/water -> unlimited/actual B -> next-day SF."""
from pathlib import Path
import shutil
import subprocess
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aquacrop_jax.canopy import calculate_potential_et
from aquacrop_jax.canopy_step import active_canopy_gdd_step
from aquacrop_jax.cuttings import apply_cutting_event, manual_cutting_schedule_step
from aquacrop_jax.fertility_daily import prepare_seeded_forage_daily, daily_fertility_no_salinity_step
from aquacrop_jax.legacy import read_project
from aquacrop_jax.potential_biomass import (PotentialBiomassParameters,
    PotentialBiomassState, potential_biomass_forage_gdd_step)
from aquacrop_jax.production import (fertility_water_productivity_step,
    dynamic_fertility_stress_step)
from aquacrop_jax.roots import canopy_water_zone
from aquacrop_jax.salinity import SaltState
from aquacrop_jax.water_crop import run_water_crop_trajectory
from test_cutting_water_feedback import initial_state, parameters, THICKNESS, FORCING

ROOT = Path(__file__).resolve().parents[1]
COEFFICIENTS = (84.93491201489957, -.5395815954288099, -.0030410169439997704)
FRACTION, TOP = .5041724311045962, 135.11909766305862


class State(NamedTuple):
    crop: object
    potential: PotentialBiomassState
    stress: jax.Array
    previous_stress: jax.Array
    top_stress: jax.Array


class Pre(NamedTuple):
    effect: object
    stress: jax.Array
    cover: jax.Array
    valid: jax.Array


class Post(NamedTuple):
    stress: jax.Array
    top_stress: jax.Array
    biomass: jax.Array
    unlimited: jax.Array
    potential_withered_before: jax.Array
    potential_withered_after: jax.Array
    harvest: object


@pytest.fixture(scope='module')
def setup():
    project = ROOT.parent / 'aquacrop-fortran/testcase/LIST/Ottawa.PRM'
    prepared = prepare_seeded_forage_daily(project, read_project(project).runs[0])
    r, p = prepared.reference.parameters, prepared.parameters
    potential = PotentialBiomassParameters(p.calendar_days, p.thermal_days,
        p.initial_cover, p.maximum_cover, r.thermal_growth, r.thermal_decline,
        r.kc_top, r.kc_decline_pct, r.late_evaporation_pct, r.minimum_gdd,
        r.water_productivity, 100., senescence_exponent=r.senescence_exponent)
    canopy = parameters()._replace(calendar_days=p.calendar_days, thermal_days=p.thermal_days,
        final_growth_gdd=p.thermal_days[2], calendar_growth=p.calendar_growth,
        calendar_decline=prepared.actual_calendar_decline)
    return prepared, potential, canopy


def simulate(theta, forcing, state, schedule, setup, *, salt=None, surface=0.,
             evaporation_depth=.15, stage2=-1.):
    actual, potential_parameters, canopy_parameters = setup

    def before(state, water, row):
        # All pre-ET inputs use the carried stress, never the pending update.
        cumulative, daily, calendar, _, eto = row
        sf = daily_fertility_no_salinity_step(state.stress, calendar - 1, True, 50,
            actual.last_virtual_day, actual.parameters, actual.record_daily_gdd, actual.first_offset)
        p = canopy_parameters._replace(calendar_days=(1, sf.full_canopy_day, 164, 164),
            thermal_days=(5, sf.full_canopy_gdd, 1803, 1803))
        zone = canopy_water_zone(water.theta, THICKNESS, jnp.zeros(3), .3, 29., 13., 46., .1)
        canopy, diag = active_canopy_gdd_step(state.crop.canopy, cumulative, daily,
            calendar, eto, zone.actual_mm, zone.field_capacity_mm, zone.wilting_point_mm, p, sf.effect)
        demand = calculate_potential_et(calendar, *p.calendar_days, canopy.canopy.cover,
            eto, 1.15, 11., .95, canopy.senescence.withered_cover, 60., 369.41, daily, 8.,
            senescence_exponent=1.)
        return (state._replace(crop=state.crop._replace(canopy=canopy)),
            demand.transpiration_mm, demand.evaporation_mm,
            Pre(sf.effect, state.stress, canopy.canopy.cover, sf.valid & diag.valid))

    def after(state, before_water, water, row):
        cumulative, daily, calendar, day, eto = row
        sf = daily_fertility_no_salinity_step(state.stress, calendar - 1, True, 50,
            actual.last_virtual_day, actual.parameters, actual.record_daily_gdd, actual.first_offset)
        potential, _ = potential_biomass_forage_gdd_step(state.potential, cumulative,
            daily, eto, 369.41, 0, potential_parameters, actual.record_daily_gdd, actual.first_offset)
        production = fertility_water_productivity_step(.15, state.crop.sum_kci,
            state.top_stress, sf.effect.reduction_water_productivity_pct, water.transpiration, eto)
        biomass = state.crop.biomass + production.biomass_increment
        updated = dynamic_fertility_stress_step(biomass, potential.biomass, FRACTION,
            *COEFFICIENTS, 50, state.previous_stress, TOP)
        cuts = state.crop.cutting._replace(interval_days=state.crop.cutting.interval_days + 1,
            interval_gdd=state.crop.cutting.interval_gdd + daily,
            potential_withered_cover=potential.maximum_withered_cover)
        decision = manual_cutting_schedule_step(state.crop.next_cut_index, day, schedule)
        canopy, cuts, harvest = apply_cutting_event(state.crop.canopy, cuts,
            decision.occurred, calendar, 25, biomass, biomass)
        potential = potential._replace(maximum_withered_cover=cuts.potential_withered_cover)
        crop = state.crop._replace(canopy=canopy, cutting=cuts, next_cut_index=decision.next_index,
            biomass=biomass, sum_kci=production.sum_kci)
        return State(crop, potential, updated.stress_pct, updated.previous_stress_pct,
            updated.sum_kc_top_stress), Post(updated.stress_pct, updated.sum_kc_top_stress,
            biomass, potential.biomass, state.potential.maximum_withered_cover,
            potential.maximum_withered_cover, harvest)

    return run_water_crop_trajectory(theta, jnp.zeros_like(forcing[4]), forcing[4],
        jnp.full_like(forcing[4], .3), None, THICKNESS, jnp.zeros(3), jnp.full(3, .29),
        46., 29., 13., 1., 1200., 46, 7., .6, .02, .01, stomatal_shape=3.,
        initial_ece=1. if salt is None else None, initial_salt=salt,
        initial_surface_water_mm=surface, initial_evap_depth_m=evaporation_depth,
        initial_stage2_start_pct=stage2, daily_feedback_step=before, initial_feedback_state=state,
        feedback_forcing=forcing, daily_post_et_step=after)


def initial():
    return State(initial_state()._replace(biomass=jnp.float64(.2)),
        PotentialBiomassState(jnp.float64(1.), jnp.float64(.7)),
        jnp.int32(50), jnp.int32(50), jnp.float64(TOP * FRACTION))


@pytest.fixture(scope='module')
def coupled(setup):
    operation = jax.jit(lambda theta, rows, state, dates: simulate(theta, rows, state, dates, setup))
    return operation(jnp.full(3, .29), FORCING, initial(), jnp.asarray([100], jnp.int32))


def test_dynamic_fertility_uses_previous_day_stress_and_cut_preserves_unlimited_mass(coupled):
    r = coupled
    np.testing.assert_array_equal(r.feedback.valid, True)
    assert int(r.feedback.stress[0]) == 50
    assert int(r.post_feedback.stress[0]) < 50
    np.testing.assert_array_equal(r.feedback.stress[1:], r.post_feedback.stress[:-1])
    np.testing.assert_allclose(r.post_feedback.top_stress,
        (1. - r.post_feedback.stress / 100.) * TOP, rtol=0., atol=1e-14)
    np.testing.assert_array_equal(r.post_feedback.harvest.occurred, [True, False, False])
    assert float(r.post_feedback.potential_withered_after[0]) == 0.
    assert float(r.post_feedback.potential_withered_before[1]) == 0.
    assert np.all(np.diff(np.asarray(r.post_feedback.unlimited)) >= 0.)
    np.testing.assert_allclose(r.water.residual, 0., rtol=0., atol=1e-9)
    np.testing.assert_allclose(r.water.salt_residual, 0., rtol=0., atol=1e-9)


def test_dynamic_fertility_scan_restart_preserves_every_carried_state(coupled, setup):
    dates = jnp.asarray([100], jnp.int32)
    operation = jax.jit(lambda theta, rows, state, salt, surface, depth, stage:
        simulate(theta, rows, state, dates, setup, salt=salt, surface=surface,
                 evaporation_depth=depth, stage2=stage))
    first = simulate(jnp.full(3, .29), tuple(v[:1] for v in FORCING), initial(), dates, setup)
    w = first.water
    second = operation(w.theta[-1], tuple(v[1:] for v in FORCING), first.final_feedback_state,
        SaltState(w.salt_dissolved[-1], w.salt_deposited[-1]), w.evap_surface_water[-1],
        w.evap_depth[-1], w.stage2_start[-1])
    for a, b in zip(jax.tree.leaves(second.final_feedback_state), jax.tree.leaves(coupled.final_feedback_state)):
        np.testing.assert_allclose(a, b, rtol=0., atol=1e-14)
    np.testing.assert_allclose(second.water.theta[-1], coupled.water.theta[-1], rtol=0., atol=1e-12)


def test_coupled_fertility_updates_match_native_observed_biomass(coupled, tmp_path):
    reference = ROOT.parent / 'aquacrop-fortran/src'
    objects = sorted(p for p in reference.glob('*.o') if p.name != 'aquacrop.o')
    if not shutil.which('gfortran') or not objects:
        pytest.skip('Requires original native objects and gfortran')
    probe = tmp_path / 'probe'
    built = subprocess.run(['gfortran', '-O0', '-I', str(reference), '-o', str(probe),
        str(ROOT / 'scripts/fertility_stress_probe.f90'), *map(str, objects)], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    rows = []
    for i, (b, u) in enumerate(zip(coupled.post_feedback.biomass, coupled.post_feedback.unlimited)):
        previous = 50 if i == 0 else int(coupled.post_feedback.stress[i - 1])
        # ETo0 submits already accumulated masses to the original stress block.
        rows.append([50, 1, float(b), float(u), FRACTION, *COEFFICIENTS, previous,
                     TOP, 0, 0, 0, 0, 1, 0, 100, 100, 0])
    text = str(len(rows)) + '\n' + '\n'.join(' '.join(format(v, '.17g') for v in r) for r in rows) + '\n'
    native = subprocess.run([str(probe)], input=text, capture_output=True, text=True)
    assert native.returncode == 0, native.stderr
    values = np.asarray([list(map(float, line.split())) for line in native.stdout.splitlines()])
    np.testing.assert_array_equal(coupled.post_feedback.stress, values[:, 0])
    np.testing.assert_allclose(coupled.post_feedback.top_stress, values[:, 4], rtol=0., atol=1e-14)

    # Verify the unlimited reference branch with its own native routine on
    # exactly the initial masses/withered state observed by these callbacks.
    potential_probe = tmp_path / 'potential-probe'
    built = subprocess.run(['gfortran', '-O0', '-I', str(reference), '-o', str(potential_probe),
        str(ROOT / 'scripts/potential_biomass_probe.f90'), *map(str, objects)],
        capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    case = tmp_path / 'case'
    shutil.copytree(reference.parent / 'testcase', case,
        ignore=shutil.ignore_patterns('OUTP', 'OUTP_REF'))
    (case / 'OUTP').mkdir()
    rows = []
    for i in range(3):
        mass = 1. if i == 0 else float(coupled.post_feedback.unlimited[i - 1])
        rows.append([int(FORCING[2][i]) - 1, float(FORCING[0][i]), float(FORCING[1][i]),
            float(FORCING[4][i]), 369.41, 0,
            float(coupled.post_feedback.potential_withered_before[i]), mass])
    text = str(len(rows)) + '\n' + '\n'.join(' '.join(format(v, '.17g') for v in r) for r in rows) + '\n'
    native = subprocess.run([str(potential_probe)], cwd=case, input=text, capture_output=True, text=True)
    assert native.returncode == 0, native.stderr
    values = np.asarray([list(map(float, line.split())) for line in native.stdout.splitlines()[1:]])
    np.testing.assert_allclose(coupled.post_feedback.unlimited, values[:, 1], rtol=0., atol=1e-14)
