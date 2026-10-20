"""Capture actual-calendar fertility effects from the original daily routine."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / 'aquacrop-fortran'


def cases():
    rows = [dict(stress_pct=s, management_stress_pct=50, virtual_time=10, germinated=True)
            for s in range(-1, 101)]
    rows.extend(dict(stress_pct=s, management_stress_pct=0, virtual_time=10, germinated=True)
                for s in (-1, 0, 50, 100))
    rows.extend(dict(stress_pct=50, management_stress_pct=50, virtual_time=v, germinated=True)
                for v in (-5, 0, 1, 163, 164))
    rows.extend(dict(stress_pct=s, management_stress_pct=50, virtual_time=10, germinated=False)
                for s in (0, 50, 100))
    rows.append(dict(stress_pct=50, management_stress_pct=-1, virtual_time=10, germinated=True))
    return rows


def generate(target_output, testcase_root=None):
    input_root = REFERENCE / 'testcase' if testcase_root is None else Path(testcase_root)
    objects = sorted(p for p in (REFERENCE / 'src').glob('*.o') if p.name != 'aquacrop.o')
    if not objects:
        raise RuntimeError('Prebuilt original native objects required')
    rows = cases()
    with tempfile.TemporaryDirectory(prefix='native-daily-fertility-') as temporary:
        work = Path(temporary)
        testcase, probe = work / 'case', work / 'probe'
        shutil.copytree(input_root, testcase,
                        ignore=shutil.ignore_patterns('OUTP', 'OUTP_REF'))
        (testcase / 'OUTP').mkdir()
        built = subprocess.run(['gfortran', '-O0', '-I', str(REFERENCE / 'src'), '-o', str(probe),
            str(ROOT / 'scripts/daily_fertility_probe.f90'), *map(str, objects)],
            capture_output=True, text=True)
        if built.returncode:
            raise RuntimeError(built.stderr)
        inputs = str(len(rows)) + '\n' + '\n'.join(
            f"{r['stress_pct']} {r['management_stress_pct']} {r['virtual_time']} {int(r['germinated'])}"
            for r in rows) + '\n'
        native = subprocess.run([str(probe)], cwd=testcase, input=inputs,
            capture_output=True, text=True)
        if native.returncode:
            raise RuntimeError(native.stderr)
        native_lines = native.stdout.splitlines()
        assert native_lines[0].startswith('crop ') and native_lines[1].startswith('thermal ')
        assert native_lines[2].startswith('record ')
        crop = list(map(float, native_lines[0].split()[1:]))
        thermal = list(map(float, native_lines[1].split()[1:]))
        record = list(map(float, native_lines[2].split()[1:]))
        for row, values in zip(rows, native_lines[3:], strict=True):
            row['expected'] = list(map(float, values.split()))
            assert len(row['expected']) == 8
    target_output.parent.mkdir(parents=True, exist_ok=True)
    target_output.write_text(json.dumps(dict(
        source='Original simul.f90::EffectSoilFertilitySalinityStress, SalinityConsidered=False, actual loaded Ottawa crop',
        crop_fields=('initial_cover', 'maximum_cover', 'calendar_growth', 'emergence_day',
            'full_canopy_day', 'senescence_day', 'maturity_day', 'flowering_day',
            'flowering_length', 'determinacy_linked', 'last_virtual_day'), crop=crop,
        thermal_fields=('emergence_gdd', 'full_canopy_gdd', 'senescence_gdd', 'maturity_gdd',
                        'shape_growth', 'shape_cover', 'shape_wp', 'shape_decline'), thermal=thermal,
        record_fields=('count', 'first_day', 'last_day', 'crop_first_day', 'gdd_method',
                       'base_temperature', 'upper_temperature'), record=record,
        output_fields=('stress_pct', 'full_canopy_day', 'full_canopy_gdd',
            'reduction_growth_pct', 'reduction_cover_pct', 'reduction_wp_pct',
            'reduction_stomatal_pct', 'canopy_decline'),
        input_sha256={path: hashlib.sha256((input_root / path).read_bytes()).hexdigest()
                      for path in ('LIST/Ottawa.PRM', 'DATA/AlfOttawaGDD.CRO', 'DATA/Ottawa.Tnx',
                                   'DATA/Ottawa.MAN', 'PARAM/Ottawa.PPn')},
        source_sha256={name: hashlib.sha256((REFERENCE / 'src' / name).read_bytes()).hexdigest()
                       for name in ('simul.f90', 'global.f90', 'tempprocessing.f90')},
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},
        cases=rows), indent=2, allow_nan=False) + '\n')
    print(f'{len(rows)} original daily fertility cases: {target_output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
        default=ROOT / 'tests/fixtures/daily_fertility_fortran73.json')
    parser.add_argument('--testcase', type=Path)
    args = parser.parse_args()
    generate(args.output, args.testcase)
