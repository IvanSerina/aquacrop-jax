"""Capture native step-3 canopy under wet soil and no pending early decline."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess

from generate_senescence_oracle import PARAMETERS, OUTPUTS, day, probe_input

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / 'aquacrop-fortran/src'


def cases():
    result = []
    def add(name, gdd, *, maximum=.95, previous=None, decline=0., cdc=.006,
            final=1920, full=600, maturity=3000, parameters=None, forcing=None):
        p = deepcopy(PARAMETERS)
        p.update(parameters or {})
        p['decline'] = cdc
        previous = maximum if previous is None else previous
        result.append(dict(name=name, parameters=p,
            canopy_state=[previous, .05, maximum, 0],
            senescence_state=[0., 0., previous, 0., 0, 0],
            forcing=forcing or [day(100., gdd)], decline=decline,
            virtual_start=179, final_growth=final, full_sf=full,
            senescence=1920, maturity=maturity))
    add('natural-onset', 1920.)
    add('natural-after-onset', math.nextafter(1920., math.inf))
    add('natural-later', 2000.)
    add('maximum-restricted-by-water-history', 2000., maximum=.4)
    add('fertility-endpoint', 2000., decline=.1)
    add('fertility-exhausted', 2000., decline=2.)
    add('fertility-reduced-maximum', 2000., maximum=.57, decline=.1,
        parameters=dict(reduction_growth_pct=24, reduction_cover_pct=40))
    add('small-maximum-below-threshold', 1920., maximum=math.nextafter(.01, 0.))
    add('small-maximum-equal-threshold', 1920., maximum=.01)
    add('zero-decline-sentinel', 1920., cdc=0.)
    add('full-cover-equals-senescence', 2000., maximum=.8, full=1920)
    add('full-cover-after-senescence', 1950., maximum=.8, full=2000)
    add('mid-persistent-maximum', 700., maximum=.8, final=500)
    add('mid-undefined-maximum', 700., maximum=-9., previous=.4, final=500)
    add('mid-fertility-ceiling', 1200., maximum=.95, final=500, decline=.5)
    add('mid-before-full-canopy', 550., maximum=.8, final=500)
    length = math.floor(((.95 + 2.29) / (.006 * 3.33)) * math.log(21.) + .50 + .5)
    add('decline-length-before', 1920. + length - 1e-6)
    add('decline-length-equal', 1920. + length)
    add('decline-length-after', 1920. + length + 1e-6)
    add('natural-persistent-trajectory', 1920., maximum=.8, decline=.1,
        forcing=[day(100., gdd, 20.) for gdd in (1920., 1940., 1960., 1980., 2000.)])
    return result


def generate():
    objects = sorted(p for p in REFERENCE.glob('*.o') if p.name != 'aquacrop.o')
    if not objects:
        raise RuntimeError('Prebuilt native objects required')
    probe = ROOT / 'build/nonexpanding-canopy-probe'
    probe.parent.mkdir(exist_ok=True)
    subprocess.run(['gcc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-O2',
        '-no-pie', '-o', str(probe), str(ROOT / 'scripts/senescence_probe.c'),
        *map(str, objects), '-l:libgfortran.so.5', '-lm'], check=True)
    rows = cases()
    for case in rows:
        lines = probe_input(case).splitlines()
        lines.insert(2, ' '.join(str(case[key]) for key in
            ('decline', 'virtual_start', 'final_growth', 'full_sf', 'senescence', 'maturity')))
        case['native_input'] = '\n'.join(lines) + '\n'
        native = subprocess.run([str(probe), '--nonexpanding'], input=case['native_input'],
            capture_output=True, text=True, check=True)
        case['expected'] = [list(map(float, line.split())) for line in native.stdout.splitlines()]
    output = ROOT / 'tests/fixtures/nonexpanding_canopy_fortran73.json'
    output.write_text(json.dumps(dict(
        source='simul.f90::DetermineCCiGDD nonexpanding canopy, wet soil, TimeSenescence=0',
        output_columns=OUTPUTS,
        source_sha256={p: hashlib.sha256((REFERENCE / p).read_bytes()).hexdigest()
                       for p in ('simul.f90', 'global.f90')},
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},
        cases=rows), indent=2, allow_nan=False) + '\n')
    print(f'{len(rows)} native cases: {output}')


if __name__ == '__main__':
    generate()
