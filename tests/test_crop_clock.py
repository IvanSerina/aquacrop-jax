"""Original germination guard, seeded local clock and distinct GDD sums."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aquacrop_jax.crop_clock import (CropClockState, accumulate_crop_gdd,
    check_germination_step, seeded_canopy_clock)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / 'tests/fixtures/germination_clock_fortran73.json').read_text())


def evaluate(row, water, daily):
    state = CropClockState(row[3].astype(jnp.bool_), row[4].astype(jnp.bool_),
        row[5].astype(jnp.int32), row[6], row[7])
    state = accumulate_crop_gdd(state, daily, *row[:3])
    clock = seeded_canopy_clock(state, *row[:2])
    state = check_germination_step(state, *row[:2], *water, row[11],
                                  seed_planted=row[12].astype(jnp.bool_))
    return clock, state


def test_germination_and_seeded_clock_match_native_eager_and_jit():
    cases = FIXTURE['cases']
    rows = jnp.asarray([c['input'] for c in cases], jnp.float64)
    water = jnp.asarray([c['root_water'] for c in cases], jnp.float64)
    heat = jnp.asarray([c['daily_gdd'] for c in cases], jnp.float64)
    expected = np.asarray([c['expected'] for c in cases])
    for operation in (jax.vmap(evaluate), jax.jit(jax.vmap(evaluate))):
        clock, state = operation(rows, water, heat)
        np.testing.assert_array_equal(np.column_stack((*clock, *state)), expected)
        assert clock.virtual_day.dtype == state.delayed_days.dtype == jnp.int32
        assert state.germinated.dtype == state.protected_seedling.dtype == jnp.bool_
        assert clock.cumulative_gdd.dtype == state.sum_gdd.dtype == jnp.float64


def test_dry_wet_sequence_captures_clock_before_germination_and_restarts_exactly():
    days = jnp.arange(100, 105, dtype=jnp.int32)
    water = jnp.asarray([0., 0., 15., 0., 0.], jnp.float64)
    initial = CropClockState(jnp.bool_(False), jnp.bool_(False), jnp.int32(0),
                             jnp.float64(0.), jnp.float64(0.))

    def step(state, forcing):
        day, actual = forcing
        state = accumulate_crop_gdd(state, 5., day, 100, 120)
        clock = seeded_canopy_clock(state, day, 100)
        state = check_germination_step(state, day, 100, actual, 30., 10., 25)
        return state, (clock, state)

    operation = jax.jit(lambda state, forcing: jax.lax.scan(step, state, forcing))
    final, (clock, states) = operation(initial, (days, water))
    np.testing.assert_array_equal(clock.virtual_day, [0, 0, 0, 1, 2])
    np.testing.assert_array_equal(clock.cumulative_gdd, [5., 5., 5., 10., 15.])
    np.testing.assert_array_equal(states.sum_gdd, [0., 0., 5., 10., 15.])
    np.testing.assert_array_equal(states.sum_gdd_from_day1, [5., 10., 15., 20., 25.])
    np.testing.assert_array_equal(states.delayed_days, [1, 2, 2, 2, 2])
    np.testing.assert_array_equal(states.protected_seedling, [False, False, True, True, True])
    mid, _ = operation(initial, (days[:2], water[:2]))
    resumed, (clock_tail, _) = operation(mid, (days[2:], water[2:]))
    for a, b in zip(resumed, final):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(clock_tail.virtual_day, clock.virtual_day[2:])
    np.testing.assert_array_equal(clock_tail.cumulative_gdd, clock.cumulative_gdd[2:])


def test_nonbinary_germination_threshold_adjacent_floats_match_native():
    for actual in (np.nextafter(16.2, -np.inf), 16.2, np.nextafter(16.2, np.inf)):
        initial = CropClockState(jnp.bool_(False), jnp.bool_(True), jnp.int32(0),
                                 jnp.float64(12.), jnp.float64(18.))
        result = jax.jit(check_germination_step)(initial, 100, 100, actual, 29., 13., 20)
        assert bool(result.germinated) == (actual >= 16.2)
        assert int(result.delayed_days) == (actual < 16.2)
        assert bool(result.protected_seedling)


def test_original_germination_clock_fixture_is_reproducible(tmp_path):
    if not shutil.which('gfortran') or not (ROOT.parent / 'aquacrop-fortran/src/simul.o').exists():
        pytest.skip('Requires original native objects and gfortran')
    output = tmp_path / 'native.json'
    native = subprocess.run([sys.executable, str(ROOT / 'scripts/generate_germination_clock_oracle.py'),
        '--output', str(output)], capture_output=True, text=True)
    assert native.returncode == 0, native.stderr
    assert json.loads(output.read_text()) == FIXTURE
