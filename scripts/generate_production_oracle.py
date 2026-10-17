"""Generate unrounded HI fixtures by linking existing immutable Fortran objects.

Run through tools/test-and-capture -- .venv/bin/python scripts/generate_production_oracle.py.
The reference sources and objects are never modified or recompiled.
"""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "aquacrop-fortran/src"


def generate():
    destination = ROOT / "build/production-probe"
    destination.parent.mkdir(exist_ok=True)
    objects = sorted(p for p in REFERENCE.glob("*.o") if p.name != "aquacrop.o")
    if not objects:
        raise RuntimeError("Prebuilt Fortran reference objects are required")
    command = ["gcc", "-O2", "-no-pie", "-o", str(destination),
               str(ROOT / "scripts/production_probe.c"), *map(str, objects), "-l:libgfortran.so.5", "-lm"]
    subprocess.run(command, check=True)
    cases = []
    for maximum, rate, flowering in [(50, 1., 10), (60, 1.5, 5), (50, .25, 10), (50, 100., 10)]:
        days = list(range(flowering - 1, flowering + int(maximum / rate + .5) + 6))
        for kind in (1, 2, 3, 4):
            text = f"{kind} {maximum} {rate:.17g} {flowering} {len(days)}\n" + " ".join(map(str, days)) + "\n"
            result = subprocess.run([str(destination)], input=text, text=True,
                                    capture_output=True, check=True)
            lines = result.stdout.splitlines()
            coefficient, switch, linear_rate = map(float, lines[0].split())
            # Keep boundary samples, including the last days before the plateau.
            duration = int(maximum / rate + .5)
            selected = {flowering + offset for offset in
                        (-1, 0, 1, 2, 3, 4, 5, int(switch) - 1, int(switch),
                         int(switch) + 1, int(switch) + 5, (int(switch) + duration) // 2,
                         *range(duration - 5, duration + 6))}
            records = []
            for line in lines[1:]:
                day, hi, lag, final = line.split()
                if int(day) in selected:
                    records.append(dict(dap=int(day), hi=float(hi), lag_pct=int(lag), final=int(final)))
            cases.append(dict(crop_file_kind=kind, maximum_hi_pct=maximum,
                daily_hi_rate_pct=rate, flowering_day=flowering,
                coefficient=coefficient, switch_day=int(switch), linear_rate=linear_rate,
                records=records))
    fixture = dict(source="AquaCrop 7.3 prebuilt objects: HarvestIndexDay, healthy canopy, no delays, seed planting",
        precision="Unrounded native float64 values; CRO codes 1..4 map to Fortran enums 0..3",
        source_sha256=hashlib.sha256((REFERENCE / "global.f90").read_bytes()).hexdigest(),
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},
        cases=cases)
    output = ROOT / "tests/fixtures/production_hi_fortran73.json"
    output.write_text(json.dumps(fixture, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    generate()
