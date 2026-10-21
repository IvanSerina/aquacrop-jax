"""Capture exact native daily-GDD/seeded-clock/germination transition order."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / 'aquacrop-fortran/src'


def generate(output):
    # Adjacent representable moisture values exercise the strict threshold.
    moistures = [0., .1, math.nextafter(.15, -math.inf), .15,
                 math.nextafter(.15, math.inf), .3]
    rows = [[day, 100, 120, germ, protected, 2, 17.5, 40., theta, 30., 10., 25, seeded, 5.]
        for day in (99, 100, 101, 121) for germ, protected, seeded in
        ((0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 0, 1)) for theta in moistures]
    rows += [[100, 100, 120, 0, 0, 0, 10., 25., theta, 29., 13., 20, 1, 0.]
             for theta in (.16199999999999998, .162, .16200000000000003)]
    source = (REFERENCE / 'run.f90').read_text()
    assignments = source.split('    else\n        ! sown or transplanted\n', 1)[1].split('        ! CC initial', 1)[0]
    assert 'VirtualTimeCC =' in assignments and 'SumGDDadjCC =' in assignments
    template = (ROOT / 'scripts/germination_clock_probe.f90').read_text()
    objects = sorted(p for p in REFERENCE.glob('*.o') if p.name != 'aquacrop.o')
    if not objects:
        raise RuntimeError('Requires original native objects')
    with tempfile.TemporaryDirectory(prefix='native-germination-') as tmp:
        work = Path(tmp)
        program, probe = work / 'probe.f90', work / 'probe'
        program.write_text(template.replace('        ! NATIVE_SEEDED_CLOCK', assignments))
        compiled = subprocess.run(['gfortran', '-O0', '-I', str(REFERENCE), '-o', str(probe),
            str(program), *map(str, objects)], capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(compiled.stderr)
        text = str(len(rows)) + '\n' + '\n'.join(' '.join(format(v, '.17g') for v in r) for r in rows) + '\n'
        native = subprocess.run([str(probe)], input=text, capture_output=True, text=True)
        if native.returncode:
            raise RuntimeError(native.stderr)
        values = [list(map(float, line.split())) for line in native.stdout.splitlines()]
    cases = []
    for row, values_row in zip(rows, values, strict=True):
        assert len(values_row) == 11
        cases.append(dict(input=row, expected=values_row[:7], root_water=values_row[7:10], daily_gdd=values_row[10]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(source='Original CheckGermination, SetGDDVariablesNextDay and verbatim seeded AdvanceOneTimeStep clock',
        input_fields=['day', 'crop_first_day', 'simulation_last_day', 'germinated', 'protected_seedling',
            'delayed_days', 'sum_gdd', 'sum_gdd_from_day1', 'theta', 'fc_pct', 'wp_pct',
            'taw_germination_pct', 'seed_planted', 'temperature_heat'],
        output_fields=['virtual_day', 'canopy_gdd_before_germination', 'germinated',
            'protected_seedling', 'delayed_days', 'sum_gdd', 'sum_gdd_from_day1'], cases=cases,
        source_sha256={name: hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest()
            for name in ('run.f90', 'simul.f90', 'global.f90')},
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects}),
        indent=2, allow_nan=False) + '\n')
    print(f'{len(cases)} original clock/germination cases: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
        default=ROOT / 'tests/fixtures/germination_clock_fortran73.json')
    generate(parser.parse_args().output)
