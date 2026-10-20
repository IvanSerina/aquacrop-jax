"""Observe first-year cuts while the immutable original routine runs a season."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / 'aquacrop-fortran'
FIELDS = ('daynr', 'count', 'last_cut_day', 'interval_days', 'cut_cover_pct',
    'senescence_gdd', 'interval_gdd', 'cumulative_gdd', 'actual_cover',
    'previous_cover', 'initial_adjusted', 'maximum_adjusted', 'withered_cover',
    'potential_withered_cover', 'cumulative_biomass', 'cumulative_yield',
    'previous_biomass_after_cut', 'previous_yield_after_cut', 'biomass_harvested', 'yield_harvested')


def generate(output):
    objects = sorted(p for p in (REFERENCE / 'src').glob('*.o') if p.name != 'aquacrop.o')
    if not objects:
        raise RuntimeError('Prebuilt immutable reference objects required')
    with tempfile.TemporaryDirectory(prefix='native-cutting-season-') as work:
        directory = Path(work)
        testcase = directory / 'case'
        shutil.copytree(REFERENCE / 'testcase', testcase,
                        ignore=shutil.ignore_patterns('OUTP', 'OUTP_REF'))
        inputs = {str(p.relative_to(testcase)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(testcase.rglob('*')) if p.is_file()}
        (testcase / 'OUTP').mkdir()
        executable = directory / 'probe'
        built = subprocess.run(['gfortran', '-O0', '-I', str(REFERENCE / 'src'), '-o', str(executable),
            str(ROOT / 'scripts/cutting_season_probe.f90'), *map(str, objects)],
            capture_output=True, text=True)
        if built.returncode:
            raise RuntimeError('Native season probe compilation failed:\n' + built.stderr)
        result = subprocess.run([str(executable)], cwd=testcase,
            capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('Native season probe failed:\n' + result.stderr)
        events = [dict(zip(FIELDS, map(float, line.split()[1:]), strict=True))
                  for line in result.stdout.splitlines() if line.startswith('cut ')]
    if not events:
        raise RuntimeError('Original first season emitted no cutting events')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(
        source='Original run.f90::AdvanceOneTimeStep on the first Ottawa.PRM season',
        source_sha256=hashlib.sha256((REFERENCE / 'src/run.f90').read_bytes()).hexdigest(),
        input_sha256=inputs,
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},
        fields=FIELDS, events=events), indent=2, allow_nan=False) + '\n')
    print(f'{len(events)} original seasonal cutting events: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
        default=ROOT / 'tests/fixtures/cutting_season_fortran73.json')
    generate(parser.parse_args().output)
