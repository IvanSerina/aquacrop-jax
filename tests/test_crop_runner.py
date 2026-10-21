from pathlib import Path
import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from aquacrop_jax.crop_runner import _crop_parameter, run_crop_stage, simulate_project
from aquacrop_jax.legacy import Project, read_project
from aquacrop_jax.legacy import read_annotated_values
from aquacrop_jax.outputs import compare_trace_to_daily

TESTCASE = Path(__file__).parents[2] / 'aquacrop-fortran/testcase'


def test_calendar_crop_uses_fortran_co2_record_as_sink_strength_alias():
    crop = read_annotated_values(TESTCASE / 'SIMUL/DEFAULT.CRO')
    assert _crop_parameter(crop, 'Sink strength') == 100.0


def test_sink_strength_still_requires_an_explicit_crop_record():
    with pytest.raises(ValueError, match='Missing crop parameter: Sink strength'):
        _crop_parameter({}, 'Sink strength')


def test_calendar_default_crop_reaches_coupled_runner(tmp_path):
    project_path = TESTCASE / 'LIST/Ottawa.PRM'
    original = read_project(project_path).runs[0]
    crop_ref = next(ref for ref in original.files if ref.role == 'Crop (CRO)')
    files = tuple(replace(ref, name='DEFAULT.CRO', directory='../SIMUL')
                  if ref is crop_ref else ref for ref in original.files)
    run = replace(original, simulation_last_day=original.simulation_first_day + 4,
                  crop_last_day=original.crop_first_day + 4, files=files)
    project = Project('calendar crop alias', 7.3, (run,))
    result, = simulate_project(project, project_path)
    assert len(result.columns['daynr']) == 5
    assert np.all(np.isfinite(result.columns['biomass']))
    np.testing.assert_allclose(result.columns['canopy'], [0., 0., 0., 0., 0.])


def test_ottawa_integrated_trace_covers_all_runs_and_balances(tmp_path):
    path = TESTCASE / 'LIST/Ottawa.PRM'
    output = tmp_path / 'coupled.csv'
    assert run_crop_stage(read_project(path), path, output) == 892
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    # SumCalendarDays(roundc(SumGDDadjCC)) can look ahead or repeat a day.
    assert [int(row['et_dap']) for row in rows[:5]] == [1, 2, 4, 5, 5]
    for name in rows[0]:
        assert np.all(np.isfinite([float(row[name]) for row in rows]))
    assert max(abs(float(row['residual'])) for row in rows) < 1e-9
    assert all(float(row['salt_residual']) == 0. for row in rows)
    assert all(float(row['salt_g_m2']) == 0. for row in rows)
    # Native AdjustYearPerennials sets RootMin=RootMax for regrowing forage.
    regrowing = [row for row in rows if int(row['run']) > 1 and int(row['dap']) > 0]
    assert regrowing
    assert all(float(row['root_depth']) == 3. for row in regrowing)
    report = compare_trace_to_daily(output, TESTCASE / 'OUTP_REF/OttawaPRMday.OUT')
    assert report['matched_days'] == 892
    assert report['missing_days'] == report['extra_days'] == []
    for name in ('rain', 'eto', 'tmin', 'tmax', 'co2', 'rootzone_ece'):
        assert report['columns'][name]['max_abs'] == 0.
    assert report['columns']['gdd']['max_abs'] <= .05000000001
    # Native daily root depth is printed to 0.01 m, including all three years.
    assert report['columns']['root_depth']['max_abs'] <= .005000000001
    # Explicitly record incomplete physics, without turning diagnostic success into parity.
    metadata = json.loads(output.with_suffix('.json').read_text())
    assert metadata['legacy_out_parity'] is False
    assert metadata['limitations']


def test_keep_swc_without_previous_run_is_rejected():
    path = TESTCASE / 'LIST/Ottawa.PRM'
    project = read_project(path)
    with pytest.raises(ValueError, match='previous simulation'):
        list(simulate_project(Project(project.description, project.version, project.runs[1:2]), path))
