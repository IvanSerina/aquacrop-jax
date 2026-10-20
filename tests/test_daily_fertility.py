"""Daily actual SF timing/effects against the original initialized crop."""
from ctypes.util import find_library
import hashlib
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

from aquacrop_jax.fertility_daily import (DailyFertilityParameters,
                                         daily_fertility_no_salinity_step)
from aquacrop_jax.kernels import degrees_day
from aquacrop_jax.legacy import read_climate_series

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / 'tests/fixtures/daily_fertility_fortran73.json').read_text())
CASES = FIXTURE['cases']


@pytest.fixture(scope='module')
def prepared():
    crop, thermal, record = FIXTURE['crop'], FIXTURE['thermal'], FIXTURE['record']
    p = DailyFertilityParameters(tuple(map(jnp.int32, crop[3:7])),
        tuple(map(jnp.int32, thermal[:4])), *map(jnp.float64, crop[:3]),
        jnp.int32(crop[7]), jnp.int32(crop[8]), jnp.bool_(crop[9]),
        tuple(map(jnp.float64, thermal[4:])))
    path = ROOT.parent / 'aquacrop-fortran/testcase/DATA/Ottawa.Tnx'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FIXTURE['input_sha256']['DATA/Ottawa.Tnx']
    temperature = np.asarray(read_climate_series(path, 2).values)
    daily = degrees_day(record[5], record[6], temperature[:, 0], temperature[:, 1], int(record[4]))
    return p, daily, jnp.int32(record[3] - record[1]), jnp.int32(crop[10])


def evaluate(parameters, daily_gdd, first_offset, last_virtual_day, rows):
    def one(row):
        stress, management, virtual, germinated = row
        return daily_fertility_no_salinity_step(stress, virtual, germinated, management,
            last_virtual_day, parameters, daily_gdd, first_offset)
    return jax.vmap(one)(rows)


def row(result):
    e = result.effect
    return jnp.stack((result.stress_pct, result.full_canopy_day, result.full_canopy_gdd,
        e.reduction_growth_pct, e.reduction_cover_pct, e.reduction_water_productivity_pct,
        e.reduction_stomatal_pct, e.canopy_decline), axis=-1)


def test_actual_daily_fertility_matches_native_eager_and_jit(prepared):
    rows = jnp.asarray([[c[k] for k in ('stress_pct', 'management_stress_pct',
        'virtual_time', 'germinated')] for c in CASES], jnp.int32)
    expected = np.asarray([c['expected'] for c in CASES])
    for operation in (evaluate, jax.jit(evaluate)):
        result = operation(*prepared, rows)
        actual = row(result)
        np.testing.assert_array_equal(actual[:, :7], expected[:, :7])
        np.testing.assert_allclose(actual[:, 7], expected[:, 7], rtol=0., atol=1e-14)
        np.testing.assert_array_equal(result.valid, True)
        assert result.full_canopy_day.dtype == jnp.int32
        assert result.full_canopy_gdd.dtype == jnp.int32
        assert result.effect.canopy_decline.dtype == jnp.float64


def test_actual_vs_reference_calendar_and_management_gdd_guard(prepared):
    p, daily, offset, last = prepared
    assert float(p.calendar_growth) == .14184615384615384
    operation = jax.jit(daily_fertility_no_salinity_step)
    active = operation(50, 10, True, 50, last, p, daily, offset)
    management_zero = operation(50, 10, True, 0, last, p, daily, offset)
    np.testing.assert_array_equal([active.full_canopy_day, active.full_canopy_gdd], [47, 590])
    np.testing.assert_array_equal([management_zero.full_canopy_day, management_zero.full_canopy_gdd], [47, 461])
    # The no-effect branch restores base thermal timing, rather than converting
    # the actual day39 with the weather record or using reference calendar40.
    for virtual, germinated, stress in ((0, True, 50), (164, True, 50),
                                       (10, False, 50), (10, True, 0)):
        result = operation(stress, virtual, germinated, 50, last, p, daily, offset)
        np.testing.assert_array_equal([result.full_canopy_day, result.full_canopy_gdd], [39, 461])
        np.testing.assert_array_equal(result.effect, np.zeros(5))
        assert int(result.stress_pct) == stress


def test_daily_fertility_scan_preserves_guard_order(prepared):
    p, daily, offset, last = prepared
    virtual = jnp.asarray([0, 1, 163, 164], jnp.int32)
    def step(previous_stress, day):
        response = daily_fertility_no_salinity_step(previous_stress, day, True,
            50, last, p, daily, offset)
        return response.stress_pct, response
    _, result = jax.jit(lambda days: lax.scan(step, jnp.int32(50), days))(virtual)
    expected = [next(c['expected'] for c in CASES if c['stress_pct'] == 50 and
        c['management_stress_pct'] == 50 and c['virtual_time'] == v and c['germinated'])
        for v in (0, 1, 163, 164)]
    np.testing.assert_array_equal(row(result)[:, :7], np.asarray(expected)[:, :7])
    np.testing.assert_allclose(row(result)[:, 7], np.asarray(expected)[:, 7], rtol=0., atol=1e-14)


def test_daily_gdd_coverage_is_only_required_when_native_conversion_runs(prepared):
    p, _, offset, last = prepared
    operation = jax.jit(daily_fertility_no_salinity_step)
    missing = jnp.empty(0, jnp.float64)
    active = operation(50, 10, True, 50, last, p, missing, offset)
    assert not bool(active.valid) and int(active.full_canopy_gdd) == 0
    no_conversion = operation(50, 10, True, 0, last, p, missing, offset)
    assert bool(no_conversion.valid) and int(no_conversion.full_canopy_gdd) == 461
    bypassed = operation(0, 10, True, 50, last, p, missing, offset)
    assert bool(bypassed.valid) and int(bypassed.full_canopy_gdd) == 461


def test_daily_fertility_fixture_is_original_and_reproducible(tmp_path):
    if (not shutil.which('gfortran') or not find_library('gfortran') or
            not (ROOT.parent / 'aquacrop-fortran/src/simul.o').exists()):
        pytest.skip('Requires immutable native objects and gfortran')
    output = tmp_path / 'native.json'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/generate_daily_fertility_oracle.py'),
        '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    native = json.loads(output.read_text())
    for key in ('crop', 'thermal', 'record', 'cases'):
        assert native[key] == FIXTURE[key]
