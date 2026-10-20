"""SumCalendarDays over complete daily files, including native EOF guards."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aquacrop_jax.kernels import degrees_day
from aquacrop_jax.thermal_time import calendar_days_from_record

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / 'tests/fixtures/calendar_days_record_fortran73.json').read_text())


@pytest.mark.parametrize('record', FIXTURE['records'], ids=lambda r: r['mode'])
def test_calendar_days_record_matches_native_eager_and_jit(record):
    meta = dict(zip(record['metadata_fields'], record['metadata'], strict=True))
    temperatures = jnp.asarray(record['temperatures'], jnp.float64)
    daily = degrees_day(meta['base_temperature'], meta['upper_temperature'],
        temperatures[:, 0], temperatures[:, 1], int(meta['gdd_method']))
    cases = record['cases']
    targets = jnp.asarray([c['target_gdd'] for c in cases], jnp.int32)
    offsets = jnp.asarray([c['first_offset'] for c in cases], jnp.int32)

    def evaluate(target, offset):
        return jax.vmap(lambda t, o: calendar_days_from_record(daily, t, o,
            undefined_year=bool(meta['undefined_year'])))(target, offset)

    for operation in (evaluate, jax.jit(evaluate)):
        actual = operation(targets, offsets)
        np.testing.assert_array_equal(actual.value, [c['expected'] for c in cases])
        np.testing.assert_array_equal(actual.valid, [c['valid'] for c in cases])
        assert actual.value.dtype == jnp.int32


def test_calendar_days_record_fixture_is_original_and_reproducible(tmp_path):
    if not shutil.which('gfortran') or not (ROOT.parent / 'aquacrop-fortran/src/tempprocessing.o').exists():
        pytest.skip('Requires original native objects and gfortran')
    output = tmp_path / 'native.json'
    result = subprocess.run([sys.executable,
        str(ROOT / 'scripts/generate_growing_degree_days_oracle.py'),
        '--calendar-days', '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text()) == FIXTURE


def test_calendar_days_scan_and_nonterminating_annual_guard():
    # The last record element is unavailable to the native loop, even when hot.
    daily = jnp.asarray([.5, .5, 100.], jnp.float64)
    targets = jnp.asarray([-1, 0, 1, 2], jnp.int32)

    def step(carry, target):
        return carry, calendar_days_from_record(daily, target)

    _, actual = jax.jit(lambda ts: jax.lax.scan(step, (), ts))(targets)
    np.testing.assert_array_equal(actual.value, [0, 0, 2, -9])
    np.testing.assert_array_equal(actual.valid, [True, True, True, False])
    cold = jax.jit(lambda d: calendar_days_from_record(d, 1, undefined_year=True))(
        jnp.zeros(365, jnp.float64))
    assert int(cold.value) == -9 and not bool(cold.valid)
    for count in (0, 1):
        for target in (0, 1):
            result = jax.jit(calendar_days_from_record)(jnp.zeros(count), target)
            assert int(result.value) == (0 if target == 0 else -9)
            assert bool(result.valid) == (target == 0)
