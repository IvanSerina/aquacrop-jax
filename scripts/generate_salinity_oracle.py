"""Run the immutable Fortran executable on isolated saline input copies.

Usage: .venv/bin/python scripts/generate_salinity_oracle.py
No source/reference output is modified. Generated fixtures retain the precision
of native day.OUT; reconstructed compartment fluxes inherit its rounding.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tempfile

from aquacrop_jax.outputs import read_daily_output

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / 'aquacrop-fortran'


def generate():
    build = ROOT / 'build'
    build.mkdir(exist_ok=True)
    cases = []
    for ksat, irrigation in [(1200., (20, 300)), (100., (20, 80))]:
        work = Path(tempfile.mkdtemp(prefix='salt-oracle-', dir=build))
        shutil.copytree(REFERENCE / 'testcase', work, dirs_exist_ok=True)
        shutil.copy2(REFERENCE / 'src/aquacrop', work / 'aquacrop')
        project = work / 'LIST/Ottawa.PRM'
        text = project.read_text().split('      2         : Year number')[0]
        text = text.replace('   (None)\n   (None)\n-- 9.', "   Salt.SW0\n   './DATA/'\n-- 9.")
        text = text.replace('   (None)\n   (None)\n-- 5.', "   Salt.IRR\n   './DATA/'\n-- 5.")
        text = text.replace("   Ottawa.OBS\n   './OBS/'", '   (None)\n   (None)')
        project.write_text(text)
        (work / 'DATA/Salt.SW0').write_text('Salt oracle\n7.3\n-9\n0\n-9\n0\n0\n0\n1\n\nThickness Water ECe\n===================\n3.05 29 8\n')
        (work / 'DATA/Salt.IRR').write_text(f'Salt oracle\n7.3\n1\n100\n1\n-9\n\nDay Depth EC\n1 {irrigation[0]} 5\n2 {irrigation[1]} 5\n3000 0 5\n')
        soil = work / 'DATA/Ottawa.SOL'
        soil.write_text(soil.read_text().replace('1200.0', f'{ksat:.1f}'))
        for name in ['Ottawa.ETo', 'Ottawa.PLU']:
            path = work / 'DATA' / name
            lines = path.read_text().splitlines()
            path.write_text('\n'.join(lines[:8] + ['0'] * (len(lines) - 8)) + '\n')
        subprocess.run([str(work / 'aquacrop')], cwd=work, check=True)
        output = read_daily_output(work / 'OUTP/OttawaPRMday.OUT')
        records = []
        for record in output.records[:3]:
            row = record.mapping()
            records.append({key: row[key] for key in ['day', 'irrigation', 'infiltration', 'drainage',
                'salt_in', 'salt_out', 'salt_total', 'root_ece', 'salt_stress',
                *[f'theta{i:02d}_pct' for i in range(1, 13)],
                *[f'ece{i:02d}' for i in range(1, 13)]]})
        cases.append(dict(ksat=ksat, initial_ece=8., irrigation_ec=5., records=records))
    fixture = dict(source='AquaCrop 7.3 native executable; isolated Ottawa with ETo=rain=0, SW0 ECe=8, IRR EC=5',
        executable_sha256=hashlib.sha256((REFERENCE / 'src/aquacrop').read_bytes()).hexdigest(),
        quantization='theta and compartment ECe: 0.1; salt ton/ha: 0.001; water mm: 0.1', cases=cases)
    destination = ROOT / 'tests/fixtures/salinity_fortran73.json'
    destination.write_text(json.dumps(fixture, indent=2) + '\n')
    print(destination)


if __name__ == '__main__':
    generate()
