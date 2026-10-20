"""Capture native early-to-natural water senescence and late rewatering."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

from generate_senescence_oracle import PARAMETERS, OUTPUTS, day, probe_input

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / 'aquacrop-fortran/src'


def cases():
    rows = []
    def add(name, *, state=None, canopy=None, parameters=None, decline=0., forcing=None):
        p = deepcopy(PARAMETERS)
        p.update(parameters or {})
        rows.append(dict(name=name, parameters=p,
            canopy_state=canopy or [.8, .05, .8, 0],
            senescence_state=state or [10., .9, .9, 0., 1, 0],
            forcing=forcing or [day(36., 1950.)], decline=decline,
            virtual_start=182, final_growth=1920, full_sf=600,
            senescence=1920, maturity=3000))
    add('ongoing-dry-late')
    add('wet-reconstructs-maximum', forcing=[day(90., 1950.)])
    add('wet-at-senescence-equality', forcing=[day(90., 1920.)])
    add('dry-at-senescence-equality', forcing=[day(36., 1920.)])
    add('no-new-late-water-senescence', state=[0., .9, .9, 7., 1, 0])
    add('dry-wet-dry-no-new-onset', forcing=[
        day(36., 1950.), day(90., 1960.), day(36., 1970.), day(36., 1980.)])
    add('cross-natural-onset', forcing=[
        day(36., 1900.), day(36., 1910.), day(36., 1920.),
        day(36., 1930.), day(90., 1940.), day(36., 1950.)])
    add('late-fertility-and-water', parameters=dict(reduction_growth_pct=24,
        reduction_cover_pct=40), canopy=[.5, .05, .5, 0],
        state=[10., .55, .55, 0., 1, 0], decline=.1,
        forcing=[day(36., 1950.), day(90., 1960.), day(36., 1970.)])
    add('pre-late-fertility-ceiling', decline=.8, forcing=[day(36., 1900.)])
    add('late-protected-rewatering', canopy=[.05, .05, .8, 1],
        forcing=[day(20., 1950.)])
    add('small-top-late-exhausted', canopy=[.0004, .0004, .0004, 0],
        state=[10., .0004, .1, 20., 1, 0], forcing=[day(20., 1950.)])
    add('late-overflow-guard', state=[1e6, .9, .9, 0., 1, 0])
    add('late-no-more-sticky', state=[10., .9, .9, 0., 1, 1],
        forcing=[day(90., 1950.)])
    add('late-topsoil-wet', forcing=[day(20., 1950., topsoil=True, top_actual=90.)])
    add('late-topsoil-dry', forcing=[day(90., 1950., topsoil=True, top_actual=36.)])
    add('late-fertility-strong-water-decline',
        parameters=dict(reduction_growth_pct=24, reduction_cover_pct=40),
        canopy=[.5, .05, .5, 0], state=[10., .55, .55, 0., 1, 0], decline=.1,
        forcing=[day(20., 1950., daily=100.)])
    return rows


def generate():
    objects = sorted(p for p in REFERENCE.glob('*.o') if p.name != 'aquacrop.o')
    if not objects:
        raise RuntimeError('Prebuilt native objects required')
    probe = ROOT / 'build/late-water-senescence-probe'
    probe.parent.mkdir(exist_ok=True)
    subprocess.run(['gcc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-O2',
        '-no-pie', '-o', str(probe), str(ROOT / 'scripts/senescence_probe.c'),
        *map(str, objects), '-l:libgfortran.so.5', '-lm'], check=True)
    cases_out = cases()
    for case in cases_out:
        lines = probe_input(case).splitlines()
        lines.insert(2, ' '.join(str(case[k]) for k in
            ('decline', 'virtual_start', 'final_growth', 'full_sf', 'senescence', 'maturity')))
        case['native_input'] = '\n'.join(lines) + '\n'
        native = subprocess.run([str(probe), '--nonexpanding'], input=case['native_input'],
            capture_output=True, text=True, check=True)
        case['expected'] = [list(map(float, line.split())) for line in native.stdout.splitlines()]
    output = ROOT / 'tests/fixtures/late_water_senescence_fortran73.json'
    output.write_text(json.dumps(dict(
        source='simul.f90::DetermineCCiGDD early/natural water senescence and rewatering',
        output_columns=OUTPUTS,
        source_sha256={p: hashlib.sha256((REFERENCE / p).read_bytes()).hexdigest()
                       for p in ('simul.f90', 'global.f90')},
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},
        cases=cases_out), indent=2, allow_nan=False) + '\n')
    print(f'{len(cases_out)} native cases: {output}')


if __name__ == '__main__':
    generate()
