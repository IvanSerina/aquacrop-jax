"""Raw legacy inputs must reproduce actual, rather than reference, timing."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aquacrop_jax.fertility_daily import (
    daily_fertility_no_salinity_step, prepare_seeded_forage_daily)
from aquacrop_jax.legacy import read_project

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT.parent / 'aquacrop-fortran/testcase'
NATIVE = json.loads((ROOT / 'tests/fixtures/daily_fertility_fortran73.json').read_text())


def prepare(case):
    project = case / 'LIST/Ottawa.PRM'
    return prepare_seeded_forage_daily(project, read_project(project).runs[0])


def assert_native(prepared, native):
    p = prepared.parameters
    np.testing.assert_array_equal(
        [p.initial_cover, p.maximum_cover, p.calendar_growth, *p.calendar_days,
         p.flowering_day, p.flowering_length, p.determinacy_linked,
         prepared.last_virtual_day], native['crop'])
    np.testing.assert_array_equal([*p.thermal_days, *p.response_shapes], native['thermal'])
    assert prepared.first_offset == native['record'][3] - native['record'][1]
    assert prepared.record_daily_gdd.shape == (int(native['record'][0]),)
    assert prepared.record_daily_gdd.dtype == jnp.float64

    def one(inputs):
        stress, management, virtual, germinated = inputs
        r = daily_fertility_no_salinity_step(stress, virtual, germinated, management,
            prepared.last_virtual_day, p, prepared.record_daily_gdd, prepared.first_offset)
        return jnp.array([r.stress_pct, r.full_canopy_day, r.full_canopy_gdd,
            r.effect.reduction_growth_pct, r.effect.reduction_cover_pct,
            r.effect.reduction_water_productivity_pct, r.effect.reduction_stomatal_pct,
            r.effect.canopy_decline]), r.valid

    inputs = jnp.asarray([[c[k] for k in ('stress_pct', 'management_stress_pct',
        'virtual_time', 'germinated')] for c in native['cases']], jnp.int32)
    actual, valid = jax.jit(jax.vmap(one))(inputs)
    expected = np.asarray([c['expected'] for c in native['cases']])
    np.testing.assert_array_equal(actual[:, :7], expected[:, :7])
    np.testing.assert_allclose(actual[:, 7], expected[:, 7], rtol=0., atol=1e-14)
    np.testing.assert_array_equal(valid, True)


def test_raw_inputs_match_native_without_native_execution_or_fixtures(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Operational adapter must not execute native processes')
    original_read = Path.read_text

    def read_text(path, *args, **kwargs):
        assert 'fixtures' not in path.parts, 'Operational adapter read an oracle'
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(subprocess, 'run', forbidden)
    monkeypatch.setattr(Path, 'read_text', read_text)
    prepared = prepare(CASE)
    assert_native(prepared, NATIVE)
    assert prepared.reference.loaded_calendar_days == (1, 39, 164, 164)
    np.testing.assert_array_equal(prepared.reference.parameters.calendar_days, [1, 40, 164, 164])
    np.testing.assert_allclose(prepared.actual_calendar_decline,
        .0027272727272727696, rtol=0., atol=1e-14)
    np.testing.assert_allclose(prepared.reference.parameters.calendar_decline,
        .030000000000000467, rtol=0., atol=1e-14)


@pytest.mark.parametrize('determinacy', [1, 2])
def test_forage_loader_resets_flowering_and_only_links_code_one(tmp_path, determinacy):
    if not shutil.which('gfortran') or not (ROOT.parent / 'aquacrop-fortran/src/simul.o').exists():
        pytest.skip('Requires original native objects and gfortran')
    case = tmp_path / 'case'
    shutil.copytree(CASE, case, ignore=shutil.ignore_patterns('OUTP', 'OUTP_REF'))
    crop = case / 'DATA/AlfOttawaGDD.CRO'
    overrides = {'Calendar Days: from sowing to flowering': 3,
        'Length of the flowering stage (days)': 2,
        'GDDays: from sowing to flowering': 30,
        'Length of the flowering stage (growing degree days)': 20,
        'Crop determinancy': determinacy}
    lines = crop.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if ':' in line:
            description = line.split(':', 1)[1]
            for prefix, value in overrides.items():
                if description.strip().startswith(prefix):
                    lines[index] = f'{value:6d} :{description}'
                    break
    crop.write_text(''.join(lines))
    output = tmp_path / 'native.json'
    result = subprocess.run([sys.executable,
        str(ROOT / 'scripts/generate_daily_fertility_oracle.py'),
        '--testcase', str(case), '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    native = json.loads(output.read_text())
    np.testing.assert_array_equal(native['crop'][7:10], [0, 0, determinacy == 1])
    prepared = prepare(case)
    assert_native(prepared, native)
    reference = prepared.reference.parameters
    assert reference.flowering_day == reference.flowering_length == 0
    assert reference.determinacy_linked == (determinacy == 1)
