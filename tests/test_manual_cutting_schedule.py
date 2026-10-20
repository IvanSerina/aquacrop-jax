"""Manual cutting selection against original full-project execution."""
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

from aquacrop_jax.cuttings import manual_cutting_schedule_step
from aquacrop_jax.legacy import read_project
from aquacrop_jax.legacy_inputs import manual_cuttings

ROOT = Path(__file__).resolve().parents[1]
TESTCASE = ROOT.parent / 'aquacrop-fortran/testcase'
ORIGINAL = json.loads((ROOT / 'tests/fixtures/cutting_season_fortran73.json').read_text())


def forecast(index, days, schedule, crop_first_day):
    def step(pointer, day):
        decision = manual_cutting_schedule_step(pointer, day, schedule, crop_first_day=crop_first_day)
        return decision.next_index, decision
    return lax.scan(step, index, days)


compiled = jax.jit(forecast)


def selected_days(testcase, events):
    selected = []
    project = read_project(testcase / 'LIST/Ottawa.PRM')
    for number, run in enumerate(project.runs, 1):
        observed = [row for row in events if int(row['run']) == number]
        first = int(observed[0]['crop_first_day']) if observed else run.crop_first_day
        schedule = manual_cuttings(testcase / 'DATA/Ottawa.MAN', first)
        days = np.arange(run.simulation_first_day, run.simulation_last_day + 1, dtype=np.int32)
        _, decisions = compiled(jnp.int32(0), jnp.asarray(days),
            jnp.asarray(schedule.day_numbers, jnp.int32), first if schedule.absolute_reference else None)
        selected.extend(days[np.asarray(decisions.occurred)].tolist())
    return selected


def test_manual_schedule_matches_eight_original_events_across_run_restarts():
    np.testing.assert_array_equal(selected_days(TESTCASE, ORIGINAL['events']),
                                  [row['daynr'] for row in ORIGINAL['events']])
    assert manual_cuttings(TESTCASE / 'DATA/Ottawa.MAN', 41414).absolute_reference


@pytest.mark.parametrize('mode', ('absolute-unsorted-overdue', 'crop-relative', 'absolute-all-expired'))
def test_manual_record_order_and_scroll_match_original_changed_inputs(mode, tmp_path):
    if (not shutil.which('gfortran') or not find_library('gfortran') or
            not (ROOT.parent / 'aquacrop-fortran/src/run.o').exists()):
        pytest.skip('Requires immutable native objects and gfortran')
    testcase = tmp_path / 'case'
    shutil.copytree(TESTCASE, testcase, ignore=shutil.ignore_patterns('OUTP', 'OUTP_REF'))
    path = testcase / 'DATA/Ottawa.MAN'
    records = path.read_text().splitlines()[:24]
    if mode == 'absolute-unsorted-overdue':
        first = read_project(testcase / 'LIST/Ottawa.PRM').runs[0].crop_first_day
        records[20] = f'{first} : absolute reference at first crop start'
        offsets = (-1, 0, 10, 10, 11, 9, 20)
    elif mode == 'crop-relative':
        records[20] = '-9 : relative crop reference'
        offsets = (2, 4)
    else:
        offsets = (0, 2)
    path.write_text('\n'.join([*records, *map(str, offsets)]) + '\n')
    output = tmp_path / 'native.json'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/generate_cutting_season_oracle.py'),
        '--testcase', str(testcase), '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    events = json.loads(output.read_text())['events']
    expected = [row['daynr'] for row in events]
    np.testing.assert_array_equal(selected_days(testcase, events), expected)
    if mode == 'absolute-unsorted-overdue':
        # Expired records are skipped, overdue/duplicate records are consumed
        # one per day. There is no sort or catch-up loop over due events.
        np.testing.assert_array_equal([row['last_cut_day'] for row in events], [10, 11, 12, 13, 20])
    elif mode == 'crop-relative':
        np.testing.assert_array_equal([row['last_cut_day'] for row in events], [2, 4] * 3)
        assert not manual_cuttings(path, 41414).absolute_reference
    else:
        assert events == []


def test_empty_disabled_and_resumed_manual_schedules():
    empty = jax.jit(manual_cutting_schedule_step)(0, 100, jnp.empty(0, jnp.int32))
    assert int(empty.next_index) == 0 and bool(empty.finished) and not bool(empty.occurred)
    disabled = jax.jit(manual_cutting_schedule_step)(0, 100, jnp.asarray([1, 2]),
        enabled=False, crop_first_day=50)
    assert int(disabled.next_index) == 0 and not bool(disabled.occurred)
    days = jnp.arange(1, 26, dtype=jnp.int32)
    schedule = jnp.asarray([-1, 0, 10, 10, 11, 9, 20], jnp.int32)
    end, whole = compiled(jnp.int32(0), days, schedule, 1)
    middle, first = compiled(jnp.int32(0), days[:11], schedule, 1)
    resumed, second = compiled(middle, days[11:], schedule, 1)
    assert int(end) == int(resumed) == len(schedule)
    np.testing.assert_array_equal(jnp.concatenate([first.occurred, second.occurred]), whole.occurred)
    np.testing.assert_array_equal(np.asarray(days)[np.asarray(whole.occurred)], [10, 11, 12, 13, 20])
