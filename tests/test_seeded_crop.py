"""Shared protection flag and native clock -> germination -> canopy order."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aquacrop_jax.canopy import EarlyVegetativeCanopyState, WaterSenescenceState
from aquacrop_jax.canopy_step import CanopyDayParameters, CanopyDayState
from aquacrop_jax.fertility import FertilityEffect
from aquacrop_jax.seeded_crop import (SeededCropState, begin_seeded_crop_day,
    germinate_seeded_crop_step, seeded_canopy_step)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / 'tests/fixtures/seeded_canopy_fortran73.json').read_text())
PARAMETERS = CanopyDayParameters((1, 60, 180, 210), (5, 600, 1920, 2200),
    1920., .05, .05, .95, .12, .012, .06, .006, .15, .55, 3., .65, 3., 40., False, 1., 10.)
EFFECT = FertilityEffect(*[jnp.float64(0.)] * 5)


def initial(germinated=False, protected=False, previous=0., sum_gdd=0., total=0., delay=0, no_more=False):
    return SeededCropState(jnp.bool_(germinated), jnp.int32(delay), jnp.float64(sum_gdd),
        jnp.float64(total), CanopyDayState(EarlyVegetativeCanopyState(jnp.float64(previous),
            jnp.float64(.05), jnp.float64(.05), jnp.bool_(protected)),
            WaterSenescenceState(jnp.float64(0.), jnp.float64(0.), jnp.float64(.05),
                jnp.float64(0.), jnp.bool_(False), jnp.bool_(no_more))))


def day_step(state, day, heat, root_water, simulation_first_day=99):
    state, captured = begin_seeded_crop_day(state, day, heat, 100, simulation_first_day, 120, 1, .05)
    state = germinate_seeded_crop_step(state, day, 100, *root_water, 25)
    state, diagnostic = seeded_canopy_step(state, captured, heat, 5., *root_water, PARAMETERS, EFFECT)
    return state, (captured, diagnostic)


def evaluate(row, water, previous, simulation_first_day, no_more):
    state = initial(row[3].astype(jnp.bool_), row[4].astype(jnp.bool_), previous,
                     row[6], row[7], row[5].astype(jnp.int32), no_more)
    return day_step(state, row[0].astype(jnp.int32), row[13], water, simulation_first_day)


def test_seeded_clock_germination_canopy_matches_native_eager_and_jit():
    rows = jnp.asarray([c['input'] for c in FIXTURE['cases']], jnp.float64)
    water = jnp.asarray([c['root_water'] for c in FIXTURE['cases']], jnp.float64)
    previous = jnp.asarray([c['previous_cover'] for c in FIXTURE['cases']], jnp.float64)
    starts = jnp.asarray([c['simulation_first_day'] for c in FIXTURE['cases']], jnp.int32)
    stopped = jnp.asarray([c['no_more'] for c in FIXTURE['cases']], jnp.bool_)
    expected_clock = np.asarray([c['expected_clock'] for c in FIXTURE['cases']])
    expected_canopy = np.asarray([c['expected_canopy'] for c in FIXTURE['cases']])
    for operation in (jax.vmap(evaluate), jax.jit(jax.vmap(evaluate))):
        state, (clock, diagnostic) = operation(rows, water, previous, starts, stopped)
        np.testing.assert_array_equal(np.column_stack((*clock, *state.clock)), expected_clock)
        canopy, sen = state.canopy
        actual = np.column_stack((*canopy[:3], *sen, diagnostic.leaf_stress_pct))
        np.testing.assert_allclose(actual[:, :7], expected_canopy[:, :7], rtol=0., atol=1e-14)
        np.testing.assert_array_equal(actual[:, 7:], expected_canopy[:, 7:])
        np.testing.assert_array_equal(diagnostic.valid, True)
        np.testing.assert_array_equal(state.clock.protected_seedling, state.canopy.canopy.protected)


def test_protection_cleared_by_canopy_stays_cleared_on_the_following_day():
    days = jnp.arange(100, 104, dtype=jnp.int32)
    heat = jnp.asarray([5., 5., 30., 5.])
    root = jnp.asarray([[10., 30., 10.], [30., 30., 10.],
                        [30., 30., 10.], [30., 30., 10.]])
    def body(state, row):
        day, daily, water = row
        state, (clock, diagnostic) = day_step(state, day, daily, water)
        return state, (state, clock, diagnostic)
    operation = jax.jit(lambda state, forcing: jax.lax.scan(body, state, forcing))
    final, (states, clock, diagnostic) = operation(initial(), (days, heat, root))
    np.testing.assert_array_equal(states.clock.protected_seedling, [False, True, False, False])
    np.testing.assert_array_equal(states.germinated, [False, True, True, True])
    np.testing.assert_array_equal(states.delayed_days, [1, 1, 1, 1])
    assert float(states.canopy.canopy.cover[2]) > 1.25 * .05
    middle, _ = operation(initial(), (days[:2], heat[:2], root[:2]))
    resumed, _ = operation(middle, (days[2:], heat[2:], root[2:]))
    for a, b in zip(jax.tree.leaves(resumed), jax.tree.leaves(final)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(diagnostic.valid, True)


def test_original_seeded_canopy_fixture_is_reproducible(tmp_path):
    if not shutil.which('gfortran') or not (ROOT.parent / 'aquacrop-fortran/src/simul.o').exists():
        pytest.skip('Requires original native objects and gfortran')
    output = tmp_path / 'native.json'
    native = subprocess.run([sys.executable, str(ROOT / 'scripts/generate_seeded_canopy_oracle.py'),
        '--output', str(output)], capture_output=True, text=True)
    assert native.returncode == 0, native.stderr
    assert json.loads(output.read_text()) == FIXTURE


def test_terminated_crop_keeps_actual_cover_on_the_initial_previous_cover_day():
    water = jnp.asarray([30., 30., 10.])
    stopped, _ = jax.jit(day_step)(initial(True, False, .0004, 40., 40., no_more=True),
                                   101, 5., water)
    assert float(stopped.canopy.canopy.cover) == .0004
    assert bool(stopped.canopy.senescence.no_more_crop)
    active, _ = jax.jit(day_step)(initial(True, False, .0004, 40., 40.), 101, 5., water)
    assert float(active.canopy.canopy.cover) > .05
    assert not bool(active.canopy.senescence.no_more_crop)
