"""One daily canopy operator against the original full-state native fixtures."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pytest

from aquacrop_jax.canopy import EarlyVegetativeCanopyState, WaterSenescenceState
from aquacrop_jax.canopy_step import (CanopyDayParameters, CanopyDayState,
                                     active_canopy_gdd_step)
from aquacrop_jax.fertility import FertilityEffect

FIXTURES = Path(__file__).parent / 'fixtures'
GROUPS = ('senescence', 'vegetative_decline', 'nonexpanding_canopy', 'late_water_senescence')


def prepare(case):
    p = case['parameters']
    c, w = case['canopy_state'], case['senescence_state']
    state = CanopyDayState(
        EarlyVegetativeCanopyState(*map(jnp.float64, c[:3]), jnp.bool_(c[3])),
        WaterSenescenceState(*map(jnp.float64, w[:4]), jnp.bool_(w[4]), jnp.bool_(w[5])))
    parameters = CanopyDayParameters(
        tuple(map(jnp.float64, (1, 60, 180, 210))),
        tuple(map(jnp.float64, (5, case.get('full_sf', 600),
                               case.get('senescence', 1920), case.get('maturity', 2200)))),
        *map(jnp.float64, (case.get('final_growth', 1920),
            p['initial_cover'], p['initial_cover'], p['maximum_cover'],
            .12, p['growth'], .06, p['decline'], .15, .55, 3.,
            p['p_senescence'], p['senescence_shape'], p['delay_eto_mm'])),
        jnp.bool_(p['fao_correction']), jnp.float64(p['adjustment_factor']), jnp.float64(p['beta_pct']))
    effect = FertilityEffect(*map(jnp.float64, (p['reduction_growth_pct'],
        p['reduction_cover_pct'], 0., case.get('decline', 0.), 0.)))
    return state, jnp.asarray(case['forcing'], jnp.float64), parameters, effect, case.get('virtual_start', 10)


def advance(state, row, p, effect, day):
    total, daily, eto, actual, fc, wp, top_actual, top_fc, top_wp, topsoil = row
    actual, fc, wp = (jnp.where(topsoil != 0., a, b) for a, b in
        ((top_actual, actual), (top_fc, fc), (top_wp, wp)))
    next_state, diagnostic = active_canopy_gdd_step(
        state, total, daily, day, eto, actual, fc, wp, p, effect)
    output = jnp.stack((*next_state.canopy, *next_state.senescence,
        diagnostic.leaf_stress_pct, diagnostic.senescence_stress_pct,
        diagnostic.senescence_threshold))
    return next_state, (output, diagnostic.valid)


def trajectory(state, forcing, p, effect, start):
    def step(carry, values):
        index, row = values
        return advance(carry, row, p, effect, index + start + 1)
    return lax.scan(step, state, (jnp.arange(forcing.shape[0]), forcing))


compiled = jax.jit(trajectory)


def assert_native(actual, expected):
    np.testing.assert_allclose(actual[:, [0, 1, 2, 5, 6]],
        expected[:, [0, 1, 2, 5, 6]], rtol=0., atol=1e-14)
    np.testing.assert_array_equal(actual[:, [3, 4, 7, 8, 9]], expected[:, [3, 4, 7, 8, 9]])
    np.testing.assert_allclose(actual[:, 10:12], expected[:, 10:12], rtol=0., atol=1e-12)
    np.testing.assert_allclose(actual[:, 12], expected[:, 12], rtol=0., atol=1e-15)


@pytest.mark.parametrize('group', GROUPS)
def test_daily_canopy_composition_matches_full_native_state(group):
    cases = json.loads((FIXTURES / f'{group}_fortran73.json').read_text())['cases']
    for case in cases:
        args = prepare(case)
        final, (actual, valid) = compiled(*args)
        if not case.get('native_finite', True):
            np.testing.assert_array_equal(valid, False)
            for before, after in zip(jax.tree.leaves(args[0]), jax.tree.leaves(final)):
                np.testing.assert_array_equal(after, before)
            assert np.all(np.isfinite(actual))
            continue
        np.testing.assert_array_equal(valid, True, err_msg=case['name'])
        assert_native(np.asarray(actual), np.asarray(case['expected']))


def test_composed_daily_state_restarts_across_natural_senescence():
    cases = json.loads((FIXTURES / 'late_water_senescence_fortran73.json').read_text())['cases']
    case = next(c for c in cases if c['name'] == 'cross-natural-onset')
    initial, forcing, p, effect, start = prepare(case)
    final, (whole, valid) = compiled(initial, forcing, p, effect, start)
    first, _ = compiled(initial, forcing[:2], p, effect, start)
    resumed, (second, _) = compiled(first, forcing[2:], p, effect, start + 2)
    np.testing.assert_array_equal(valid, True)
    np.testing.assert_allclose(second, whole[2:], rtol=0., atol=1e-14)
    for a, b in zip(jax.tree.leaves(final), jax.tree.leaves(resumed)):
        np.testing.assert_allclose(a, b, rtol=0., atol=1e-14)
    # One eager step also verifies the public scalar call outside scan/JIT.
    _, (eager, valid) = advance(initial, forcing[0], p, effect, start + 1)
    assert bool(valid)
    assert_native(np.asarray(eager)[None, :], np.asarray(case['expected'])[:1])


def test_active_canopy_rejects_days_outside_its_phase_without_changing_state():
    case = json.loads((FIXTURES / 'late_water_senescence_fortran73.json').read_text())['cases'][0]
    initial, _, p, effect, _ = prepare(case)
    operation = jax.jit(active_canopy_gdd_step)
    for total in (4., 3001.):
        final, diagnostic = operation(initial, total, 10., 190., 5., 100., 100., 20., p, effect)
        assert not bool(diagnostic.valid)
        for a, b in zip(jax.tree.leaves(final), jax.tree.leaves(initial)):
            np.testing.assert_array_equal(a, b)
