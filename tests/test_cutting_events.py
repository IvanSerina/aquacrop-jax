"""Cut application matches the unmodified native block, including strict guards."""
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

from aquacrop_jax.canopy import EarlyVegetativeCanopyState, WaterSenescenceState
from aquacrop_jax.canopy_step import CanopyDayState
from aquacrop_jax.cuttings import CuttingState, apply_cutting_event

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / 'tests/fixtures/cutting_event_fortran73.json').read_text())


def initial(p):
    canopy = CanopyDayState(EarlyVegetativeCanopyState(
        *map(jnp.float64, (p['previous_cover'], p['initial_adjusted'], p['maximum_adjusted'])),
        jnp.bool_(True)), WaterSenescenceState(jnp.float64(17.), jnp.float64(.88),
        jnp.float64(p['withered_cover']), jnp.float64(13.5), jnp.bool_(True), jnp.bool_(True)))
    cuts = CuttingState(*[jnp.int32(p[k]) for k in ('count', 'last_cut_day', 'interval_days')],
        *[jnp.float64(p[k]) for k in ('interval_gdd', 'previous_biomass',
                                     'previous_yield', 'potential_withered_cover')])
    return canopy, cuts


def test_cutting_application_matches_native_exactly_eager_and_jit():
    for case in FIXTURE['cases']:
        p = case['input']
        canopy, cuts = initial(p)
        args = (canopy, cuts, p['occurred'], p['day_in_season'],
                p['cut_cover_pct'], p['cumulative_biomass'], p['cumulative_yield'])
        for operation in (apply_cutting_event, jax.jit(apply_cutting_event)):
            next_canopy, next_cuts, record = operation(*args)
            actual = [next_cuts.count, next_cuts.last_cut_day, next_cuts.interval_days,
                next_cuts.interval_gdd, next_canopy.canopy.cover, p['actual_cover'],
                next_canopy.canopy.initial_adjusted, next_canopy.canopy.maximum_adjusted,
                next_canopy.senescence.withered_cover, next_cuts.potential_withered_cover,
                next_cuts.previous_biomass, next_cuts.previous_yield,
                p['cumulative_biomass'], p['cumulative_yield'],
                record.biomass_harvested, record.yield_harvested]
            np.testing.assert_array_equal(actual, case['expected'], err_msg=case['name'])
            assert bool(record.occurred) == bool(p['occurred'])
            assert bool(record.canopy_reduced) == (case['expected'][4] != p['previous_cover'])
            assert int(record.interval_days) == p['interval_days']
            # The event does not reset these native canopy/senescence fields.
            np.testing.assert_array_equal(next_canopy.canopy.protected, canopy.canopy.protected)
            for index in (0, 1, 3, 4, 5):
                np.testing.assert_array_equal(next_canopy.senescence[index], canopy.senescence[index])
            assert next_cuts.count.dtype == jnp.int32
            assert next_cuts.previous_biomass.dtype == jnp.float64


def test_native_cutting_fixture_can_be_regenerated_from_original_block(tmp_path):
    if (not shutil.which('gfortran') or not find_library('gfortran') or
            not (ROOT.parent / 'aquacrop-fortran/src/run.o').exists()):
        pytest.skip('Requires immutable native objects and gfortran')
    output = tmp_path / 'native.json'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/generate_cutting_event_oracle.py'),
        '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    native = json.loads(output.read_text())
    assert native['block_sha256'] == FIXTURE['block_sha256']
    assert native['cases'] == FIXTURE['cases']


def sequence(canopy, cuts, forcing):
    def step(state, row):
        previous_canopy, previous_cuts = state
        occurred, day, cover, biomass, yield_mass = row
        # Counters advance before schedule evaluation in AdvanceOneTimeStep.
        previous_cuts = previous_cuts._replace(
            interval_days=previous_cuts.interval_days + 1,
            interval_gdd=previous_cuts.interval_gdd + 10.)
        current_canopy = previous_canopy._replace(canopy=previous_canopy.canopy._replace(
            cover=cover, maximum_adjusted=cover))
        next_canopy, next_cuts, record = apply_cutting_event(current_canopy,
            previous_cuts, occurred, day, 25, biomass, yield_mass)
        return (next_canopy, next_cuts), record
    return lax.scan(step, (canopy, cuts), forcing)


def test_cutting_scan_records_before_reset_and_restarts():
    p = FIXTURE['cases'][0]['input'] | dict(count=0, last_cut_day=-9,
        interval_days=0, interval_gdd=0., previous_biomass=0., previous_yield=0.)
    canopy, cuts = initial(p)
    forcing = jnp.asarray([[0, 1, .8, 2., 1.], [1, 2, .7, 3., 1.5],
                           [0, 3, .4, 5., 2.], [1, 4, .6, 6., 3.]], jnp.float64)
    operation = jax.jit(sequence)
    final, records = operation(canopy, cuts, forcing)
    np.testing.assert_array_equal(records.occurred, [False, True, False, True])
    np.testing.assert_array_equal(records.interval_days, [1, 2, 1, 2])
    np.testing.assert_array_equal(records.count, [0, 1, 1, 2])
    np.testing.assert_array_equal(records.biomass_harvested, [0., 3., 0., 3.])
    np.testing.assert_array_equal(records.yield_harvested, [0., 1.5, 0., 1.5])
    assert float(jnp.sum(records.biomass_harvested)) == 6.
    assert float(jnp.sum(records.yield_harvested)) == 3.
    assert float(final[0].canopy.cover) == .25
    assert int(final[1].interval_days) == 0
    assert float(final[1].interval_gdd) == 0.
    first, _ = operation(canopy, cuts, forcing[:2])
    resumed, _ = operation(*first, forcing[2:])
    for a, b in zip(jax.tree.leaves(final), jax.tree.leaves(resumed)):
        np.testing.assert_array_equal(a, b)
