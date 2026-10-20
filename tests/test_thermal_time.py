"""Unrounded daily record -> sequential GDD -> exact native integer result."""
from ctypes.util import find_library
import json
from pathlib import Path
import shutil
import subprocess
import sys

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pytest

from aquacrop_jax.kernels import degrees_day
from aquacrop_jax.thermal_time import growing_degree_days_from_record

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / 'tests/fixtures/growing_degree_days_fortran73.json').read_text())


def evaluate(temperature, periods, offsets, base, upper, method, undefined):
    daily = degrees_day(base, upper, temperature[:, 0], temperature[:, 1], method)
    return jax.vmap(lambda period, offset: growing_degree_days_from_record(
        daily, period, offset, undefined_year=undefined))(periods, offsets)


@pytest.mark.parametrize('record', FIXTURE['records'], ids=lambda r: r['mode'])
def test_record_gdd_matches_original_eager_and_jit(record):
    meta = dict(zip(record['metadata_fields'], record['metadata'], strict=True))
    cases = record['cases']
    args = (jnp.asarray(record['temperatures'], jnp.float64),
        jnp.asarray([c['period_days'] for c in cases], jnp.int32),
        jnp.asarray([c['first_offset'] for c in cases], jnp.int32),
        meta['base_temperature'], meta['upper_temperature'], int(meta['gdd_method']),
        bool(meta['undefined_year']))
    for operation in (evaluate, jax.jit(evaluate)):
        actual = operation(*args)
        np.testing.assert_array_equal(actual.value, [c['expected'] for c in cases])
        np.testing.assert_array_equal(actual.valid, [c['valid'] for c in cases])
        assert actual.value.dtype == jnp.int32


def test_original_thermal_time_fixture_is_reproducible(tmp_path):
    if (not shutil.which('gfortran') or not find_library('gfortran') or
            not (ROOT.parent / 'aquacrop-fortran/src/tempprocessing.o').exists()):
        pytest.skip('Requires original native objects and gfortran')
    output = tmp_path / 'native.json'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/generate_growing_degree_days_oracle.py'),
        '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())['records'] == FIXTURE['records']


def test_sequential_half_integer_and_native_roundc_neighbourhood():
    record = next(r for r in FIXTURE['records'] if r['mode'] == 'half-sum')
    case = next(c for c in record['cases'] if c['period_days'] == 10 and c['first_offset'] == 0)
    # 0.15 accumulated chronologically is 1.4999999999999998. The distance
    # equals epsilon, so roundc's strict near-half guard does not apply.
    assert case['expected'] == 1
    result = jax.jit(growing_degree_days_from_record)(jnp.full(11, .15), 10, 0)
    assert int(result.value) == 1 and bool(result.valid)
    half = next(r for r in FIXTURE['records'] if r['mode'] == 'half-round')
    expected = [next(c['expected'] for c in half['cases']
        if c['first_offset'] == 0 and c['period_days'] == n) for n in range(1, 6)]
    np.testing.assert_array_equal(expected, [0, 1, 2, 2, 2])


def test_record_gdd_scan_distinguishes_valid_cold_days_from_missing_start():
    daily = jnp.zeros(5, jnp.float64)
    periods = jnp.asarray([1, 1, 5, 0], jnp.int32)
    offsets = jnp.asarray([0, 4, 0, 0], jnp.int32)
    def step(carry, forcing):
        period, offset = forcing
        value = growing_degree_days_from_record(daily, period, offset)
        return carry, value
    _, result = jax.jit(lambda forcing: lax.scan(step, (), forcing))((periods, offsets))
    np.testing.assert_array_equal(result.value, [0, 0, -9, -9])
    np.testing.assert_array_equal(result.valid, [True, False, False, False])
    for count in (0, 1):
        empty = jax.jit(growing_degree_days_from_record)(jnp.zeros(count), 1)
        assert int(empty.value) == 0 and not bool(empty.valid)
