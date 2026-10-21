"""Unlimited potential biomass against initialized native forage recurrence."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aquacrop_jax.fertility_daily import prepare_seeded_forage_daily
from aquacrop_jax.legacy import read_project, read_annotated_values
from aquacrop_jax.potential_biomass import (PotentialBiomassParameters,
    PotentialBiomassState, potential_biomass_forage_gdd_step)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / 'tests/fixtures/potential_biomass_fortran73.json').read_text())


@pytest.fixture(scope='module')
def prepared():
    project = ROOT.parent / 'aquacrop-fortran/testcase/LIST/Ottawa.PRM'
    actual = prepare_seeded_forage_daily(project, read_project(project).runs[0])
    r = actual.reference.parameters
    crop = read_annotated_values(project.parent.parent / 'DATA/AlfOttawaGDD.CRO')
    sink = next(v for key, v in crop.items() if key.startswith('Sink strength'))
    p = PotentialBiomassParameters(actual.parameters.calendar_days, actual.parameters.thermal_days,
        r.initial_cover, r.maximum_cover, r.thermal_growth, r.thermal_decline,
        r.kc_top, r.kc_decline_pct, r.late_evaporation_pct, r.minimum_gdd,
        r.water_productivity, sink, senescence_exponent=r.senescence_exponent)
    np.testing.assert_array_equal([*p.calendar_days, *p.thermal_days, *p[2:]], FIXTURE['parameters'])
    return p, actual.record_daily_gdd, actual.first_offset


def step(state, row, prepared):
    _, heat, daily, eto, co2, delay, _, _ = row
    p, record, offset = prepared
    return potential_biomass_forage_gdd_step(state, heat, daily, eto, co2, delay, p, record, offset)


def test_potential_biomass_matches_native_independent_calls_eager_and_jit(prepared):
    rows = jnp.asarray([c['input'] for c in FIXTURE['cases']], jnp.float64)
    expected = np.asarray([c['expected'] for c in FIXTURE['cases']])
    def one(row):
        return step(PotentialBiomassState(row[7], row[6]), row, prepared)
    for operation in (jax.vmap(one), jax.jit(jax.vmap(one))):
        state, diagnostics = operation(rows)
        np.testing.assert_allclose(np.column_stack((state.maximum_withered_cover, state.biomass)),
            expected, rtol=0., atol=1e-14)
        np.testing.assert_array_equal(diagnostics.valid, True)
        assert state.biomass.dtype == jnp.float64
        assert diagnostics.thermal_calendar_day.dtype == jnp.int32
        zero_eto = np.asarray(rows[:, 3]) == 0.
        np.testing.assert_array_equal(state.biomass[zero_eto], rows[zero_eto, 7])
        np.testing.assert_array_equal(diagnostics.increment[zero_eto], 0.)
        # Native CO2 correction is disabled when rounded cents match CO2Ref.
        same_cents = np.asarray(rows[:, 4]) == 369.414
        np.testing.assert_array_equal(diagnostics.water_productivity[same_cents], .15)


def test_potential_biomass_scan_and_restart_match_native_persistent_state(prepared):
    rows = jnp.asarray([c['input'] for c in FIXTURE['episode']], jnp.float64)
    expected = np.asarray([c['expected'] for c in FIXTURE['episode']])
    initial = PotentialBiomassState(jnp.float64(0.), jnp.float64(0.))
    def body(state, row):
        next_state, diagnostics = step(state, row, prepared)
        return next_state, (next_state, diagnostics)
    operation = jax.jit(lambda state, rs: jax.lax.scan(body, state, rs))
    final, (states, diagnostics) = operation(initial, rows)
    np.testing.assert_allclose(np.column_stack((states.maximum_withered_cover, states.biomass)),
        expected, rtol=0., atol=1e-14)
    middle, first = operation(initial, rows[:4])
    resumed, second = operation(middle, rows[4:])
    for actual, expected_leaf in zip(jax.tree_util.tree_leaves(resumed), jax.tree_util.tree_leaves(final)):
        np.testing.assert_array_equal(actual, expected_leaf)
    for a, b, whole in zip(jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(second),
                           jax.tree_util.tree_leaves((states, diagnostics))):
        np.testing.assert_array_equal(jnp.concatenate((a, b)), whole)
    assert float(states.maximum_withered_cover[3]) > float(states.maximum_withered_cover[2])
    assert float(states.biomass[3]) == float(states.biomass[2])


def test_potential_biomass_missing_climate_preserves_state(prepared):
    p, record, _ = prepared
    initial = PotentialBiomassState(jnp.float64(2.), jnp.float64(.3))
    result, diagnostics = jax.jit(potential_biomass_forage_gdd_step)(
        initial, 500., 10., 4., 401., 0, p, record, len(record))
    np.testing.assert_array_equal(result, initial)
    assert not bool(diagnostics.valid)


def test_original_potential_biomass_fixture_is_reproducible(tmp_path):
    if not shutil.which('gfortran') or not (ROOT.parent / 'aquacrop-fortran/src/simul.o').exists():
        pytest.skip('Requires original native objects and gfortran')
    output = tmp_path / 'native.json'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/generate_potential_biomass_oracle.py'),
        '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text()) == FIXTURE
