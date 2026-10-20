"""Actual vegetative canopy after SF full cover, without water senescence."""
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
    early_vegetative_gdd_step, fertility_canopy_ceiling_gdd, leaf_thresholds_for_eto)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / 'tests/fixtures/vegetative_decline_fortran73.json').read_text())


def trajectory(initial, forcing, parameters, decline, virtual_start):
    p = parameters
    def step(state, values):
        index, row = values
        total, daily, eto, actual, fc, wp, top_actual, top_fc, top_wp, topsoil = row
        water, fc, wp = (jnp.where(topsoil != 0., a, b) for a, b in
            ((top_actual, actual), (top_fc, fc), (top_wp, wp)))
        upper, lower = leaf_thresholds_for_eto(eto, .15, .55,
            p['fao_correction'], p['adjustment_factor'])
        cap = fertility_canopy_ceiling_gdd(index + virtual_start + 1, total,
            FIXTURE['calendar_days'], FIXTURE['thermal_days'],
            p['initial_cover'], p['maximum_cover'], .12, p['growth'], .06, p['decline'],
            p['reduction_growth_pct'], p['reduction_cover_pct'], decline)
        next_state, leaf, valid = early_vegetative_gdd_step(state, total, daily,
            5., 600., 1920., 1920., 2200., p['initial_cover'], p['maximum_cover'],
            p['growth'], p['decline'], p['reduction_growth_pct'],
            p['reduction_cover_pct'], water, fc, wp, upper, lower, 3.,
            fertility_decline_cap=cap)
        return next_state, (jnp.stack((*next_state, leaf)), valid)
    return lax.scan(step, initial, (jnp.arange(forcing.shape[0]), forcing))


compiled_trajectory = jax.jit(trajectory)


@pytest.mark.parametrize('case', FIXTURE['cases'], ids=lambda c: c['name'])
def test_vegetative_decline_matches_native_eager_and_jit(case):
    values = case['canopy_state']
    initial = EarlyVegetativeCanopyState(*map(jnp.float64, values[:3]), jnp.bool_(values[3]))
    parameters = {k: jnp.asarray(v) for k, v in case['parameters'].items()}
    args = (initial, jnp.asarray(case['forcing'], jnp.float64), parameters,
            case['decline'], case['virtual_start'])
    expected = np.asarray(case['expected'])[:, [0, 1, 2, 3, 10]]
    for operation in (trajectory, compiled_trajectory):
        _, (actual, valid) = operation(*args)
        np.testing.assert_array_equal(valid, True)
        np.testing.assert_allclose(actual[:, :3], expected[:, :3], rtol=0., atol=1e-14)
        np.testing.assert_array_equal(actual[:, 3], expected[:, 3])
        np.testing.assert_allclose(actual[:, 4], expected[:, 4], rtol=0., atol=1e-12)
        assert actual.dtype == jnp.float64


def test_vegetative_fixture_still_matches_original_objects(tmp_path):
    reference = ROOT.parent / 'aquacrop-fortran/src'
    objects = sorted(p for p in reference.glob('*.o') if p.name != 'aquacrop.o')
    if not shutil.which('gcc') or not find_library('gfortran') or not objects:
        pytest.skip('Requires native reference objects and gcc')
    probe = tmp_path / 'probe'
    subprocess.run(['gcc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-O2',
        '-no-pie', '-o', str(probe), str(ROOT / 'scripts/senescence_probe.c'),
        *map(str, objects), '-l:libgfortran.so.5', '-lm'], check=True, capture_output=True)
    for case in FIXTURE['cases']:
        result = subprocess.run([str(probe), '--fertility-decline'],
            input=case['native_input'], capture_output=True, text=True, check=True)
        actual = [list(map(float, line.split())) for line in result.stdout.splitlines()]
        np.testing.assert_array_equal(actual, case['expected'])


def test_decline_cap_preserves_early_guard_and_rejects_natural_senescence():
    state = EarlyVegetativeCanopyState(jnp.float64(.5), jnp.float64(.05),
        jnp.float64(.5), jnp.bool_(False))
    args = (state, 700., 10., 5., 600., 1920., 1920., 2200.,
            .05, .95, .012, .006, 0., 0., 90., 100., 20., .15, .55, 3.)
    _, _, valid = early_vegetative_gdd_step(*args)
    assert not bool(valid)
    args = (state, 1920., *args[2:])
    _, _, valid = early_vegetative_gdd_step(*args, fertility_decline_cap=.9)
    assert not bool(valid)


def test_ceiling_freezes_at_senescence_endpoint_under_jit():
    operation = jax.jit(lambda gdd: fertility_canopy_ceiling_gdd(181., gdd,
        FIXTURE['calendar_days'], FIXTURE['thermal_days'], .05, .95,
        .12, .012, .06, .006, 0., 0., .1))
    endpoint = .95 - ((180. - 60.) / (1920. - 600.) * .1 / 100.) * (1920. - 600.)
    np.testing.assert_allclose([operation(1920.), operation(2000.)],
                               endpoint, rtol=0., atol=1e-14)


@pytest.mark.parametrize('case', [c for c in FIXTURE['cases'] if c['name'].startswith('early-')],
                         ids=lambda c: c['name'])
def test_second_inverse_also_applies_without_extended_phase(case):
    p = case['parameters']
    state = EarlyVegetativeCanopyState(*map(jnp.float64, case['canopy_state'][:3]), jnp.bool_(False))
    total, daily, eto, actual, fc, wp = case['forcing'][0][:6]
    upper, lower = leaf_thresholds_for_eto(eto, .15, .55)
    result, _, valid = jax.jit(early_vegetative_gdd_step)(state, total, daily,
        5., 600., 1920., 1920., 2200., .05, .95, .012, .006,
        p['reduction_growth_pct'], p['reduction_cover_pct'], actual, fc, wp,
        upper, lower, 3.)
    assert bool(valid)
    np.testing.assert_allclose(result[:3], np.asarray(case['expected'])[0, :3], rtol=0., atol=1e-14)
