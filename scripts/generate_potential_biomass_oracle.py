"""Observe unlimited biomass using original daily Fortran routine and state."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / 'aquacrop-fortran'


def generate(output):
    rows = [[v, g, d, eto, co2, delay, withered, 2.] for v, g, d in
        [(0, 0., 0.), (1, 5., 5.), (10, 100., 10.), (40, 500., 10.),
         (100, 1200., 15.), (163, 1803., 1.), (164, 1804., 10.)]
        for eto, co2, delay, withered in [(4., 369.41, 0, 0.), (0., 401., 3, 0.),
            (4., 550., 0, .99), (4., 369.414, 0, 0.), (4., 369.416, 0, 0.)]]
    episode = [[i, g, d, eto, 401., 2, 0., 0.] for i, g, d, eto in
        [(0, 0., 0., 4.), (1, 6., 6., 4.), (2, 20., 14., 4.),
         (3, 100., 80., 0.), (4, 500., 10., 4.), (5, 1200., 10., 4.),
         (6, 1800., 10., 4.), (7, 1803., 3., 4.), (8, 1804., 1., 4.)]]
    objects = sorted(p for p in (REFERENCE / 'src').glob('*.o') if p.name != 'aquacrop.o')
    if not objects:
        raise RuntimeError('Requires original native objects')
    with tempfile.TemporaryDirectory(prefix='native-potential-biomass-') as tmp:
        work = Path(tmp)
        case, probe = work / 'case', work / 'probe'
        shutil.copytree(REFERENCE / 'testcase', case,
            ignore=shutil.ignore_patterns('OUTP', 'OUTP_REF'))
        (case / 'OUTP').mkdir()
        built = subprocess.run(['gfortran', '-O0', '-I', str(REFERENCE / 'src'), '-o', str(probe),
            str(ROOT / 'scripts/potential_biomass_probe.f90'), *map(str, objects)],
            capture_output=True, text=True)
        if built.returncode:
            raise RuntimeError(built.stderr)
        results = []
        for inputs, flags in ((rows, []), (episode, ['--scan'])):
            text = str(len(inputs)) + '\n' + '\n'.join(' '.join(map(str, r)) for r in inputs) + '\n'
            native = subprocess.run([str(probe), *flags], cwd=case, input=text, capture_output=True, text=True)
            if native.returncode:
                raise RuntimeError(native.stderr)
            lines = [list(map(float, line.split())) for line in native.stdout.splitlines()]
            header = lines[0]
            results.append([dict(input=r, expected=e) for r, e in zip(inputs, lines[1:], strict=True)])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(
        source='Original simul.f90::DeterminePotentialBiomass, initialized Ottawa forage/GDD crop',
        parameter_fields=['emergence_day', 'full_canopy_day', 'senescence_day', 'maturity_day',
            'emergence_gdd', 'full_canopy_gdd', 'senescence_gdd', 'maturity_gdd',
            'initial_cover', 'maximum_cover', 'thermal_growth', 'thermal_decline',
            'kc_top', 'kc_decline_pct', 'late_evaporation_pct', 'minimum_gdd',
            'water_productivity', 'sink_strength_pct', 'wet_bare_coefficient', 'senescence_exponent'],
        parameters=header, input_fields=['virtual_time', 'cumulative_gdd', 'daily_gdd',
            'eto', 'co2', 'delayed_days', 'initial_withered', 'initial_biomass'],
        output_fields=['maximum_withered_cover', 'biomass'], cases=results[0], episode=results[1],
        input_sha256={name: hashlib.sha256((REFERENCE / 'testcase' / name).read_bytes()).hexdigest()
            for name in ('DATA/AlfOttawaGDD.CRO', 'DATA/Ottawa.Tnx', 'PARAM/Ottawa.PPn')},
        source_sha256={name: hashlib.sha256((REFERENCE / 'src' / name).read_bytes()).hexdigest()
            for name in ('simul.f90', 'global.f90')},
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects}),
        indent=2, allow_nan=False) + '\n')
    print(f'{len(rows)} independent calls, {len(episode)} native scan days: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
        default=ROOT / 'tests/fixtures/potential_biomass_fortran73.json')
    generate(parser.parse_args().output)
