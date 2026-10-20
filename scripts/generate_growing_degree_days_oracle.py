"""Capture original daily-file GrowingDegreeDays on finite and annual records."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from aquacrop_jax.legacy import read_climate_series

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / 'aquacrop-fortran'


def generate(output):
    objects = sorted(p for p in (REFERENCE / 'src').glob('*.o') if p.name != 'aquacrop.o')
    if not objects:
        raise RuntimeError('Prebuilt immutable native objects required')
    records = []
    with tempfile.TemporaryDirectory(prefix='native-growing-gdd-') as temporary:
        work = Path(temporary)
        probe = work / 'probe'
        built = subprocess.run(['gfortran', '-O0', '-I', str(REFERENCE / 'src'), '-o', str(probe),
            str(ROOT / 'scripts/growing_degree_days_probe.f90'), *map(str, objects)],
            capture_output=True, text=True)
        if built.returncode:
            raise RuntimeError(built.stderr)
        for mode in ('finite', 'annual-undefined', 'half-sum', 'half-round'):
            testcase = work / mode
            (testcase / 'DATA').mkdir(parents=True)
            temperature = testcase / 'DATA/Ottawa.Tnx'
            shutil.copyfile(REFERENCE / 'testcase/DATA/Ottawa.Tnx', temperature)
            if mode == 'annual-undefined':
                lines = temperature.read_text().splitlines()
                lines[4] = '1901 : undefined reference year'
                temperature.write_text('\n'.join(lines[:8 + 365]) + '\n')
            elif mode in ('half-sum', 'half-round'):
                lines = temperature.read_text().splitlines()[:8]
                rows = ['0 .3'] * 11 if mode == 'half-sum' else ['0 1'] * 6
                temperature.write_text('\n'.join([*lines, *rows]) + '\n')
            series = read_climate_series(temperature, 2)
            count = len(series.values)
            periods = sorted(set([-1, 0, *range(1, 21), 39, 47, 164, 365, 730, count - 1, count, count + 1]))
            offsets = [0, count - 2, count - 1]
            if count > 140:
                offsets.insert(1, 140)
            if mode != 'annual-undefined':
                offsets = [-1, *offsets, count]
            inputs = [(period, offset) for offset in offsets for period in periods]
            text = str(len(inputs)) + '\n' + '\n'.join(f'{period} {offset}'
                for period, offset in inputs) + '\n'
            command = [str(probe)] + (['--base-zero'] if mode.startswith('half-') else [])
            native = subprocess.run(command, cwd=testcase, input=text,
                capture_output=True, text=True)
            if native.returncode:
                raise RuntimeError(mode + ': ' + native.stderr)
            native_rows = [line.split() for line in native.stdout.splitlines() if line.strip()]
            metadata = list(map(float, native_rows[0]))
            assert int(metadata[0]) == count
            assert len(native_rows[1:]) == len(inputs)
            cyclic = bool(metadata[5])
            assert cyclic == (mode == 'annual-undefined')
            cases = []
            for (period, offset), row in zip(inputs, native_rows[1:], strict=True):
                values = list(map(int, row))
                assert values[:2] == [period, offset]
                valid = period > 0 and 0 <= offset < count - 1 and (cyclic or offset + period <= count - 1)
                cases.append(dict(period_days=period, first_offset=offset,
                    expected=values[2], valid=valid))
            records.append(dict(mode=mode, metadata_fields=('count', 'record_first_day',
                'record_last_day', 'example_first_day', 'gdd_method', 'undefined_year',
                'base_temperature', 'upper_temperature', 'default_tmin', 'default_tmax'),
                metadata=metadata, temperatures=series.values,
                input_sha256=hashlib.sha256(temperature.read_bytes()).hexdigest(), cases=cases))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(
        source='Original daily loaders and tempprocessing.f90::GrowingDegreeDays, controlled method3/Tupper30 and Tbase0/5',
        source_sha256={name: hashlib.sha256((REFERENCE / 'src' / name).read_bytes()).hexdigest()
                       for name in ('tempprocessing.f90', 'global.f90')},
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},
        records=records), indent=2, allow_nan=False) + '\n')
    print(f'{sum(len(record["cases"]) for record in records)} original GDD cases: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
        default=ROOT / 'tests/fixtures/growing_degree_days_fortran73.json')
    generate(parser.parse_args().output)
