"""Native mid/late canopy state before water senescence and final guards."""
from ctypes.util import find_library
import json
from pathlib import Path
import shutil
import subprocess

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pytest

from aquacrop_jax.canopy import (EarlyVegetativeCanopyState,
    fertility_canopy_ceiling_gdd, nonexpanding_canopy_gdd_step)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / 'tests/fixtures/nonexpanding_canopy_fortran73.json').read_text())


def trajectory(initial, forcing, p, decline, start, final, full, senescence, maturity):
    calendar = (1, 60, 180, 210)
    thermal = (5, full, senescence, maturity)
    ratio = jnp.where(full < senescence, 120. / jnp.where(full < senescence,
        senescence - full, 1.), 1.)
    stress_maximum = p['maximum_cover'] * (1. - p['reduction_cover_pct'] / 100.)
    def step(state, values):
        index, row = values
        total = row[0]
        cap = fertility_canopy_ceiling_gdd(index + start + 1, total, calendar, thermal,
            p['initial_cover'], p['maximum_cover'], .12, p['growth'], .06, p['decline'],
            p['reduction_growth_pct'], p['reduction_cover_pct'], decline)
        canopy, leaf, valid = nonexpanding_canopy_gdd_step(state, total, final,
            full, senescence, maturity, p['maximum_cover'], stress_maximum,
            p['decline'], cap, decline, ratio)
        return canopy, (jnp.stack((*canopy, leaf)), valid)
    return lax.scan(step, initial, (jnp.arange(forcing.shape[0]), forcing))


compiled_trajectory = jax.jit(trajectory)


@pytest.mark.parametrize('case', FIXTURE['cases'], ids=lambda c: c['name'])
def test_nonexpanding_canopy_matches_native_state_and_jit(case):
    values = case['canopy_state']
    initial = EarlyVegetativeCanopyState(*map(jnp.float64, values[:3]), jnp.bool_(values[3]))
    p = {k: jnp.asarray(v) for k, v in case['parameters'].items()}
    args = (initial, jnp.asarray(case['forcing'], jnp.float64), p,
            *(case[k] for k in ('decline', 'virtual_start', 'final_growth',
                                'full_sf', 'senescence', 'maturity')))
    expected = np.asarray(case['expected'])[:, [0, 1, 2, 3, 10]]
    for operation in (trajectory, compiled_trajectory):
        _, (actual, valid) = operation(*args)
        np.testing.assert_array_equal(valid, True)
        np.testing.assert_allclose(actual[:, :3], expected[:, :3], rtol=0., atol=1e-14)
        np.testing.assert_array_equal(actual[:, 3:], expected[:, 3:])
        assert actual.dtype == jnp.float64


def test_native_nonexpanding_fixture_is_reproducible(tmp_path):
    reference = ROOT.parent / 'aquacrop-fortran/src'
    objects = sorted(p for p in reference.glob('*.o') if p.name != 'aquacrop.o')
    if not shutil.which('gcc') or not find_library('gfortran') or not objects:
        pytest.skip('Requires immutable native objects and gcc')
    probe = tmp_path / 'probe'
    subprocess.run(['gcc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-O2',
        '-no-pie', '-o', str(probe), str(ROOT / 'scripts/senescence_probe.c'),
        *map(str, objects), '-l:libgfortran.so.5', '-lm'], check=True, capture_output=True)
    for case in FIXTURE['cases']:
        native = subprocess.run([str(probe), '--nonexpanding'], input=case['native_input'],
            capture_output=True, text=True, check=True)
        actual = [list(map(float, line.split())) for line in native.stdout.splitlines()]
        np.testing.assert_array_equal(actual, case['expected'])


def test_nonexpanding_guard_preserves_state_outside_phase():
    state = EarlyVegetativeCanopyState(jnp.float64(.8), jnp.float64(.05),
        jnp.float64(.8), jnp.bool_(False))
    operation = jax.jit(lambda total: nonexpanding_canopy_gdd_step(state,
        total, 1920., 600., 1920., 2200., .95, .95, .006, .95, 0., 120. / 1320.))
    for total in (1919., 2201.):
        result, _, valid = operation(total)
        assert not bool(valid)
        np.testing.assert_array_equal(result, state)
    # roundc, rather than a direct floating-point comparison, controls maturity.
    _, _, valid = operation(2200.49)
    assert bool(valid)
