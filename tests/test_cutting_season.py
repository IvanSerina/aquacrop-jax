"""Check event application and records against the original Ottawa daily loop."""
from ctypes.util import find_library
import json
from pathlib import Path
import shutil
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aquacrop_jax.canopy import EarlyVegetativeCanopyState, WaterSenescenceState
from aquacrop_jax.canopy_step import CanopyDayState
from aquacrop_jax.climate import date_to_day_number
from aquacrop_jax.cuttings import CuttingState, apply_cutting_event
from aquacrop_jax.legacy import read_project
from aquacrop_jax.legacy_inputs import manual_cuttings
from datetime import date

ROOT = Path(__file__).resolve().parents[1]
TESTCASE = ROOT.parent / 'aquacrop-fortran/testcase'
FIXTURE = json.loads((ROOT / 'tests/fixtures/cutting_season_fortran73.json').read_text())


def test_original_daily_loop_cuts_match_manual_schedule_and_legacy_records():
    run = read_project(TESTCASE / 'LIST/Ottawa.PRM').runs[0]
    schedule = manual_cuttings(TESTCASE / 'DATA/Ottawa.MAN', run.crop_first_day)
    expected_days = [day for day in schedule.day_numbers
                     if run.simulation_first_day <= day <= run.simulation_last_day]
    events = FIXTURE['events']
    np.testing.assert_array_equal([row['daynr'] for row in events], expected_days)
    legacy = []
    for line in (TESTCASE / 'OUTP_REF/OttawaPRMharvests.OUT').read_text().splitlines():
        fields = line.split()
        if len(fields) < 10:
            continue
        try:
            number, day, month, year, dap, interval = map(int, fields[:6])
        except ValueError:
            continue
        if year == 2014 and 0 < number < 9999:
            legacy.append((date_to_day_number(date(year, month, day)), number, dap, interval,
                           *map(float, fields[6:10])))
    assert len(events) == len(legacy) == 2
    for event, record in zip(events, legacy, strict=True):
        np.testing.assert_array_equal([event[k] for k in
            ('daynr', 'count', 'last_cut_day', 'interval_days')], record[:4])
        # RecordHarvest prints these mass fields to .3 decimals.
        np.testing.assert_allclose([event[k] for k in ('biomass_harvested',
            'cumulative_biomass', 'yield_harvested', 'cumulative_yield')],
            record[4:], rtol=0., atol=.000500000001)
        assert event['actual_cover'] > event['cut_cover_pct'] / 100.
        assert event['previous_cover'] == event['maximum_adjusted'] == .25
        assert event['withered_cover'] == event['potential_withered_cover'] == 0.
        assert event['previous_biomass_after_cut'] == event['cumulative_biomass']
        assert event['previous_yield_after_cut'] == event['cumulative_yield']


def test_cutting_records_match_raw_original_season_masses():
    previous_biomass, previous_yield, previous_day = 0., 0., -9
    operation = jax.jit(apply_cutting_event)
    for event in FIXTURE['events']:
        # Observe event inputs/outputs only. This does not feed oracle values
        # into a seasonal JAX simulation or reconstruct unobserved water state.
        canopy = CanopyDayState(EarlyVegetativeCanopyState(
            jnp.float64(event['actual_cover']), jnp.float64(event['initial_adjusted']),
            jnp.float64(event['actual_cover']), jnp.bool_(False)),
            WaterSenescenceState(jnp.float64(0.), jnp.float64(0.),
                jnp.float64(event['actual_cover']), jnp.float64(0.), jnp.bool_(False), jnp.bool_(False)))
        state = CuttingState(jnp.int32(event['count'] - 1), jnp.int32(previous_day),
            jnp.int32(event['interval_days']), jnp.float64(event['interval_gdd']),
            jnp.float64(previous_biomass), jnp.float64(previous_yield), jnp.float64(1.))
        next_canopy, next_state, record = operation(canopy, state, True,
            int(event['last_cut_day']), int(event['cut_cover_pct']),
            event['cumulative_biomass'], event['cumulative_yield'])
        np.testing.assert_array_equal([record.biomass_harvested, record.yield_harvested,
            next_canopy.canopy.cover, next_canopy.canopy.maximum_adjusted,
            next_state.previous_biomass, next_state.previous_yield],
            [event[k] for k in ('biomass_harvested', 'yield_harvested', 'previous_cover',
                'maximum_adjusted', 'previous_biomass_after_cut', 'previous_yield_after_cut')])
        previous_biomass, previous_yield = event['cumulative_biomass'], event['cumulative_yield']
        previous_day = event['last_cut_day']


def test_original_seasonal_cutting_probe_is_reproducible(tmp_path):
    if (not shutil.which('gfortran') or not find_library('gfortran') or
            not (ROOT.parent / 'aquacrop-fortran/src/run.o').exists()):
        pytest.skip('Requires original objects and gfortran')
    output = tmp_path / 'native-season.json'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/generate_cutting_season_oracle.py'),
        '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    native = json.loads(output.read_text())
    assert native['events'] == FIXTURE['events']
