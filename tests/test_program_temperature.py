"""Project PP temperature controls follow the executable's positional loader."""
from ctypes.util import find_library
import csv
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from aquacrop_jax.legacy_inputs import program_temperature_parameters
from aquacrop_jax.climate import run_climate_stage
from aquacrop_jax.crop_runner import simulate_project
from aquacrop_jax.legacy import read_project

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT.parent / 'aquacrop-fortran/testcase/PARAM/Ottawa.PPn'


@pytest.fixture(scope='module')
def native_probe(tmp_path_factory):
    reference = ROOT.parent / 'aquacrop-fortran/src'
    objects = sorted(p for p in reference.glob('*.o') if p.name != 'aquacrop.o')
    if not shutil.which('gfortran') or not find_library('gfortran') or not objects:
        pytest.skip('Requires native reference objects and gfortran')
    probe = tmp_path_factory.mktemp('program-temperature') / 'probe'
    built = subprocess.run(['gfortran', '-O0', '-I', str(reference), '-o', str(probe),
        str(ROOT / 'scripts/program_temperature_probe.f90'), *map(str, objects)],
        capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    return probe


def test_positional_cli_parameters_and_clamps_match_native(native_probe, tmp_path):
    project = tmp_path / 'LIST/case.PRM'
    program = tmp_path / 'PARAM/case.PPn'
    program.parent.mkdir()
    for method in (-9, 0, 1, 2, 3, 4, 127):
        # Translated comments and D notation must not change parser behavior.
        records = [line.split(':', 1)[0] + ' : descrizione' for line in TEMPLATE.read_text().splitlines()]
        records[9] = '2.5D0 : senescenza'
        records[20] = f'{method} : metodo'
        program.write_text('\n'.join(records) + '\n')
        native = subprocess.run([str(native_probe), str(program)],
            capture_output=True, text=True, check=True)
        expected = list(map(float, native.stdout.split()))
        actual = program_temperature_parameters(project)
        np.testing.assert_array_equal([actual.gdd_method, actual.senescence_exponent], expected)


def test_project_type_selects_pp1_and_missing_file_uses_defaults(native_probe, tmp_path):
    project = tmp_path / 'LIST/case.PRO'
    program = tmp_path / 'PARAM/case.PP1'
    program.parent.mkdir()
    program.write_text(TEMPLATE.read_text())
    # A multi-project file with different values must not override PP1.
    wrong = program.with_suffix('.PPn')
    records = TEMPLATE.read_text().splitlines()
    records[9] = '2.5'
    records[20] = '1'
    wrong.write_text('\n'.join(records) + '\n')
    actual = program_temperature_parameters(project)
    assert (actual.gdd_method, actual.senescence_exponent) == (3, 1.)
    program.unlink()
    native = subprocess.run([str(native_probe), str(program)],
        capture_output=True, text=True, check=True)
    actual = program_temperature_parameters(project)
    np.testing.assert_array_equal([actual.gdd_method, actual.senescence_exponent],
                                  list(map(float, native.stdout.split())))


def test_truncated_parameter_file_is_rejected(tmp_path):
    program = tmp_path / 'PARAM/case.PPn'
    program.parent.mkdir()
    program.write_text('4\n1.1\n')
    with pytest.raises(ValueError, match='25 CLI records'):
        program_temperature_parameters(tmp_path / 'LIST/case.PRM')


def test_changed_project_method_reaches_climate_and_crop_runners(native_probe, tmp_path):
    testcase = tmp_path / 'case'
    shutil.copytree(TEMPLATE.parent.parent, testcase,
                    ignore=shutil.ignore_patterns('OUTP', 'OUTP_REF'))
    project_file = testcase / 'LIST/Ottawa.PRM'
    program = testcase / 'PARAM/Ottawa.PPn'
    records = program.read_text().splitlines()
    records[9], records[20] = '2.5', '2'
    program.write_text('\n'.join(records) + '\n')
    project = read_project(project_file)
    project = replace(project, runs=project.runs[:1])
    output = tmp_path / 'climate.csv'
    assert run_climate_stage(project, project_file, output) > 0
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    # Same real temperatures and project control are fed to the native
    # DegreesDay, rather than using an oracle from a different method.
    native = subprocess.run([str(native_probe), str(program), '--gdd'],
        input=''.join(f"5 30 {r['tmin']} {r['tmax']}\n" for r in rows),
        capture_output=True, text=True, check=True)
    expected = np.asarray([float(v) for v in native.stdout.splitlines()[1:]])
    assert int(native.stdout.splitlines()[0].split()[0]) == 2
    np.testing.assert_allclose([float(r['gdd']) for r in rows], expected, rtol=0., atol=1e-12)
    crop = next(simulate_project(project, project_file))
    np.testing.assert_allclose(crop.columns['gdd'], expected, rtol=0., atol=1e-12)
    np.testing.assert_allclose(crop.columns['residual'], 0., rtol=0., atol=1e-9)
    np.testing.assert_allclose(crop.columns['salt_residual'], 0., rtol=0., atol=1e-9)
