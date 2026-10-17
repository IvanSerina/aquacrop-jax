"""Native rounding near half-integers and integer bounds, without recompiling Fortran."""
from pathlib import Path
import hashlib
import json
import math
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "aquacrop-fortran/src"


def generate():
    probe = ROOT / "build/rounding-probe"
    probe.parent.mkdir(exist_ok=True)
    objects = sorted(p for p in REFERENCE.glob("*.o") if p.name != "aquacrop.o")
    if not objects:
        raise RuntimeError("Prebuilt Fortran reference objects required")
    subprocess.run(["gcc", "-O2", "-no-pie", "-o", str(probe),
        str(ROOT / "scripts/rounding_probe.c"), *map(str, objects),
        "-l:libgfortran.so.5", "-lm"], check=True)
    values = {0., 1e-20, -1e-20, -1e20, 1e20}
    for half in [-128.5, -127.5, -9.5, -8.5, -2.5, -1.5, -.5,
                  .5, 1.5, 2.5, 12.5, 120.5, 126.5, 127.5]:
        values.add(half)
        for direction in (-math.inf, math.inf):
            neighbour = half
            for _ in range(4):
                neighbour = math.nextafter(neighbour, direction)
                values.add(neighbour)
    for bound in [-2147483648., 2147483647., -128., 127.]:
        values.update([bound - 1., bound, bound + 1.])
    values = sorted(values)
    text = str(len(values)) + "\n" + "\n".join(map(repr, values)) + "\n"
    result = subprocess.run([str(probe)], input=text, text=True, capture_output=True, check=True)
    records = [dict(value=v, int32=int(line.split()[0]), int8=int(line.split()[1]))
               for v, line in zip(values, result.stdout.splitlines(), strict=True)]
    output = ROOT / "tests/fixtures/rounding_fortran73.json"
    output.write_text(json.dumps(dict(source="Native ac_utils::roundc_int32 and roundc_int8",
        source_sha256=hashlib.sha256((REFERENCE / "utils.f90").read_bytes()).hexdigest(),
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},
        records=records), indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    generate()
