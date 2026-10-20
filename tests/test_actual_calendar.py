"""Compare actual-climate CDC against the unchanged original native routine."""
import hashlib
from pathlib import Path
import shutil
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aquacrop_jax.actual_calendar import calendar_decline_from_record
from aquacrop_jax.kernels import degrees_day
from aquacrop_jax.legacy import read_climate_series

ROOT = Path(__file__).resolve().parents[1]
ROWS = [(.95, .006, 1803, 1803, 164, 140),  # actual Ottawa, fallback 5 GDD
        (.95, .006, 461, 3000, 39, 140),     # complete decline
        (.95, .006, 461, 590, 39, 140),      # partial decline
        (.95, .006, 590, 461, 47, 140),      # senescence after maturity
        (.6, .003, 100, 1000, 10, 140),
        (.6, .003, 100, 120, 10, 140),
        (.1, .02, 5, 100, 1, 140),
        (.95, 0., 461, 3000, 39, 140),
        (.95, .006, 461, 590, 39, -100),     # uncovered lookup start
        (.95, .006, 461, 590, 39, 1056),     # starts at excluded final record
        (.95, .006, 461, 3000, 39, 1055)]    # starts covered, record exhausted


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    reference = ROOT.parent / 'aquacrop-fortran'
    objects = sorted(p for p in (reference / 'src').glob('*.o') if p.name != 'aquacrop.o')
    if not shutil.which('gfortran') or not objects:
        pytest.skip('Requires original native objects and gfortran')
    work = tmp_path_factory.mktemp('actual-calendar')
    (work / 'DATA').mkdir()
    source = reference / 'testcase/DATA/Ottawa.Tnx'
    shutil.copyfile(source, work / 'DATA/Ottawa.Tnx')
    probe = work / 'probe'
    built = subprocess.run(['gfortran', '-O0', '-I', str(reference / 'src'), '-o', str(probe),
        str(ROOT / 'scripts/calendar_decline_probe.f90'), *map(str, objects)],
        capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    inputs = str(len(ROWS)) + '\n' + '\n'.join(' '.join(map(str, r)) for r in ROWS) + '\n'
    native = subprocess.run([str(probe)], cwd=work, input=inputs, capture_output=True, text=True)
    assert native.returncode == 0, native.stderr
    expected = np.asarray([float(line) for line in native.stdout.splitlines()])
    temperature = jnp.asarray(read_climate_series(source, 2).values, jnp.float64)
    daily = degrees_day(5., 30., temperature[:, 0], temperature[:, 1], 3)
    assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256((work / 'DATA/Ottawa.Tnx').read_bytes()).digest()
    return daily, expected


def test_actual_calendar_decline_matches_original_eager_jit_and_scan(oracle):
    daily, expected = oracle
    rows = jnp.asarray(ROWS, jnp.float64)

    def one(row):
        return calendar_decline_from_record(daily, *row)

    scan = lambda rs: jax.lax.scan(lambda carry, r: (carry, one(r)), (), rs)[1]
    for operation in (jax.vmap(one), jax.jit(jax.vmap(one)), jax.jit(scan)):
        result = operation(rows)
        np.testing.assert_allclose(result.coefficient, expected, rtol=0., atol=1e-14)
        np.testing.assert_array_equal(result.valid, expected != -9.)
        assert result.decline_gdd.dtype == result.decline_days.dtype == jnp.int32
        assert result.coefficient.dtype == jnp.float64
        assert int(result.decline_gdd[0]) == 5
        assert int(result.decline_days[0]) == 11
        np.testing.assert_allclose(result.coefficient[0], .0027272727272727696, rtol=0., atol=1e-14)


def test_actual_calendar_decline_rejects_zero_canopy_and_cold_annual_record():
    for maximum, daily in ((0., jnp.ones(365)), (.95, jnp.zeros(365))):
        result = jax.jit(lambda d: calendar_decline_from_record(
            d, maximum, .006, 100, 100, 10, undefined_year=True))(daily)
        assert not bool(result.valid) and float(result.coefficient) == -9.
