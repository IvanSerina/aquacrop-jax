"""Native vegetative canopy growth with the fertility-decline ceiling."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

from generate_senescence_oracle import PARAMETERS, OUTPUTS, day, probe_input

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / 'aquacrop-fortran/src'


def cases():
    result = []
    def add(name, canopy=(.95, .05, .95, 0), *, decline=1., start=69,
            parameters=None, forcing=None):
        p = deepcopy(PARAMETERS)
        p.update(parameters or {})
        result.append(dict(name=name, parameters=p, canopy_state=list(canopy),
            senescence_state=[0., 0., canopy[0], 0., 0, 0],
            forcing=forcing or [day(90., 650.)], decline=decline, virtual_start=start))
    add('full-cover-declines')
    add('water-limited-expansion', (.4, .05, .4, 0), forcing=[day(60., 650.)])
    add('wet-expansion', (.4, .05, .4, 0))
    add('maximum-near-boundary', (.97999 * .95, .05, .95, 0))
    add('first-growth-bypasses-ceiling', (.04, .04, .04, 0), decline=1000.)
    add('protected-first-growth', (.05, .05, .05, 1))
    add('decline-zero', decline=0.)
    add('decline-threshold-equal', decline=1e-6)
    add('decline-threshold-above', decline=1.000001e-6)
    add('calendar-before-full', start=49)
    add('thermal-full-equal', forcing=[day(90., 600.)])
    add('ceiling-zero', decline=1000.)
    add('fertility-growth-cover', (.5, .05, .5, 0), decline=.3,
        parameters=dict(reduction_growth_pct=24, reduction_cover_pct=40))
    add('fertility-cover-reduced-dry', (.4, .05, .4, 0), decline=.3,
        parameters=dict(reduction_growth_pct=24, reduction_cover_pct=40),
        forcing=[day(60., 650.)])
    add('persistent-decline-rewatering', (.6, .05, .6, 0), decline=2., forcing=[
        day(water, 650. + 20. * i, 20.) for i, water in enumerate([60., 90., 60., 90., 90.])])
    add('early-water-limited-inverse', (.4, .05, .4, 0), start=19,
        forcing=[day(52., 100.)])
    add('early-fertility-inverse', (.4, .05, .4, 0), start=19,
        parameters=dict(reduction_growth_pct=24, reduction_cover_pct=40),
        forcing=[day(60., 100.)])
    return result


def native_input(case):
    lines = probe_input(case).splitlines()
    lines.insert(2, f"{case['decline']:.17g} {case['virtual_start']}")
    return '\n'.join(lines) + '\n'


def generate():
    objects = sorted(p for p in REFERENCE.glob('*.o') if p.name != 'aquacrop.o')
    if not objects:
        raise RuntimeError('Prebuilt native objects required')
    probe = ROOT / 'build/vegetative-decline-probe'
    probe.parent.mkdir(exist_ok=True)
    subprocess.run(['gcc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-O2',
        '-no-pie', '-o', str(probe), str(ROOT / 'scripts/senescence_probe.c'),
        *map(str, objects), '-l:libgfortran.so.5', '-lm'], check=True)
    rows = cases()
    for case in rows:
        case['native_input'] = native_input(case)
        native = subprocess.run([str(probe), '--fertility-decline'],
            input=case['native_input'], capture_output=True, text=True, check=True)
        case['expected'] = [list(map(float, line.split())) for line in native.stdout.splitlines()]
    output = ROOT / 'tests/fixtures/vegetative_decline_fortran73.json'
    output.write_text(json.dumps(dict(
        source='simul.f90::DetermineCCiGDD vegetative growth and CCxSFCD ceiling',
        output_columns=OUTPUTS, calendar_days=[1, 60, 180, 210],
        thermal_days=[5, 600, 1920, 2200],
        source_sha256={p: hashlib.sha256((REFERENCE / p).read_bytes()).hexdigest()
                       for p in ('simul.f90', 'global.f90')},
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},
        cases=rows), indent=2, allow_nan=False) + '\n')
    print(f'{len(rows)} native cases: {output}')


if __name__ == '__main__':
    generate()
