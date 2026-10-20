"""Full native canopy state through natural onset, drought and rewatering."""
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

from aquacrop_jax.canopy import (EarlyVegetativeCanopyState, WaterSenescenceState,
    LateCanopyParameters, early_vegetative_gdd_step, early_water_senescence_gdd_step,
    fertility_canopy_ceiling_gdd, leaf_thresholds_for_eto,
    nonexpanding_canopy_gdd_step, senescence_threshold_for_eto)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / 'tests/fixtures/late_water_senescence_fortran73.json').read_text())


def trajectory(initial, forcing, p, decline, start):
    stress_max = p['maximum_cover'] * (1. - p['reduction_cover_pct'] / 100.)
    late = LateCanopyParameters(jnp.float64(3000.), p['growth'],
        p['reduction_growth_pct'], p['reduction_cover_pct'])
    def step(carry, values):
        canopy, water_state = carry
        index, row = values
        total, daily, eto, actual, fc, wp, top_actual, top_fc, top_wp, topsoil = row
        water, fc, wp = (jnp.where(topsoil != 0., a, b) for a, b in
            ((top_actual, actual), (top_fc, fc), (top_wp, wp)))
        upper, lower = leaf_thresholds_for_eto(eto, .15, .55,
            p['fao_correction'], p['adjustment_factor'])
        cap = fertility_canopy_ceiling_gdd(index + start + 1, total,
            (1, 60, 180, 210), (5, 600, 1920, 3000),
            p['initial_cover'], p['maximum_cover'], .12, p['growth'], .06, p['decline'],
            p['reduction_growth_pct'], p['reduction_cover_pct'], decline)
        developing, leaf, valid = lax.cond(total < 1920.,
            lambda _: early_vegetative_gdd_step(canopy, total, daily,
                5., 600., 1920., 1920., 3000., p['initial_cover'], p['maximum_cover'],
                p['growth'], p['decline'], p['reduction_growth_pct'],
                p['reduction_cover_pct'], water, fc, wp, upper, lower, 3.,
                fertility_decline_cap=cap),
            lambda _: nonexpanding_canopy_gdd_step(canopy, total,
                1920., 600., 1920., 3000., p['maximum_cover'], stress_max,
                p['decline'], cap, decline, 120. / 1320.), None)
        threshold = senescence_threshold_for_eto(eto, p['p_senescence'], water_state.time_gdd,
            p['fao_correction'], p['adjustment_factor'], p['beta_pct'])
        result, next_water, stress, valid = early_water_senescence_gdd_step(
            water_state, developing, canopy.cover, total, daily,
            5., 1920., 1920., p['initial_cover'], p['initial_cover'],
            p['maximum_cover'], stress_max, p['decline'], water, fc, wp,
            eto, p['p_senescence'], p['senescence_shape'], p['delay_eto_mm'],
            canopy_valid=valid, fao_correction=p['fao_correction'],
            adjustment_factor=p['adjustment_factor'], beta_pct=p['beta_pct'],
            fertility_decline_cap=cap, late_canopy=late)
        return (result, next_water), (jnp.stack((*result, *next_water, leaf, stress, threshold)), valid)
    return lax.scan(step, initial, (jnp.arange(forcing.shape[0]), forcing))


compiled_trajectory = jax.jit(trajectory)


@pytest.mark.parametrize('case', FIXTURE['cases'], ids=lambda c: c['name'])
def test_late_water_senescence_matches_native_eager_and_jit(case):
    c, w = case['canopy_state'], case['senescence_state']
    initial = (EarlyVegetativeCanopyState(*map(jnp.float64, c[:3]), jnp.bool_(c[3])),
        WaterSenescenceState(*map(jnp.float64, w[:4]), jnp.bool_(w[4]), jnp.bool_(w[5])))
    p = {k: jnp.asarray(v) for k, v in case['parameters'].items()}
    args = (initial, jnp.asarray(case['forcing'], jnp.float64), p,
            case['decline'], case['virtual_start'])
    expected = np.asarray(case['expected'])
    for operation in (trajectory, compiled_trajectory):
        _, (actual, valid) = operation(*args)
        np.testing.assert_array_equal(valid, True)
        np.testing.assert_allclose(actual[:, [0, 1, 2, 5, 6]],
            expected[:, [0, 1, 2, 5, 6]], rtol=0., atol=1e-14)
        np.testing.assert_array_equal(actual[:, [3, 4, 7, 8, 9]], expected[:, [3, 4, 7, 8, 9]])
        np.testing.assert_allclose(actual[:, 10:12], expected[:, 10:12], rtol=0., atol=1e-12)
        np.testing.assert_allclose(actual[:, 12], expected[:, 12], rtol=0., atol=1e-15)
        assert np.all(np.isfinite(actual))


def test_late_fixture_matches_original_native_objects(tmp_path):
    reference = ROOT.parent / 'aquacrop-fortran/src'
    objects = sorted(p for p in reference.glob('*.o') if p.name != 'aquacrop.o')
    if not shutil.which('gcc') or not find_library('gfortran') or not objects:
        pytest.skip('Requires original objects and gcc')
    probe = tmp_path / 'probe'
    subprocess.run(['gcc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-O2',
        '-no-pie', '-o', str(probe), str(ROOT / 'scripts/senescence_probe.c'),
        *map(str, objects), '-l:libgfortran.so.5', '-lm'], check=True, capture_output=True)
    for case in FIXTURE['cases']:
        native = subprocess.run([str(probe), '--nonexpanding'], input=case['native_input'],
            capture_output=True, text=True, check=True)
        actual = [list(map(float, line.split())) for line in native.stdout.splitlines()]
        np.testing.assert_array_equal(actual, case['expected'])
