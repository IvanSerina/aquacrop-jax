"""Capture soil fertility response outputs from the immutable native objects."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "aquacrop-fortran/src"
SHAPES = [0., 1e-6, .001, .01, .0499999999999999, .05,
          .0500000000000001, .1, .25, .5, 1., 2., 5., 10., 25.]
STRESSES = [0, 1, 10, 20, 50, 75, 99, 100, -1, 101]


def generate():
    probe = ROOT / "build/fertility-probe"
    probe.parent.mkdir(exist_ok=True)
    objects = sorted(p for p in REFERENCE.glob("*.o") if p.name != "aquacrop.o")
    if not objects:
        raise RuntimeError("Prebuilt Fortran reference objects required")
    subprocess.run(["gcc", "-O2", "-no-pie", "-o", str(probe),
        str(ROOT / "scripts/fertility_probe.c"), *map(str, objects),
        "-l:libgfortran.so.5", "-lm"], check=True)
    cases = []
    for stress in STRESSES:
        for index in range(len(SHAPES)):
            shapes = [SHAPES[(index + offset * 3) % len(SHAPES)] for offset in range(4)]
            cases.append(dict(stress_level_pct=stress, shape_growth=shapes[0],
                shape_cover=shapes[1], shape_water_productivity=shapes[2],
                shape_canopy_decline=shapes[3]))
    text = str(len(cases)) + "\n" + "\n".join(
        " ".join(str(c[k]) for k in ("stress_level_pct", "shape_growth", "shape_cover",
                                    "shape_water_productivity", "shape_canopy_decline"))
        for c in cases) + "\n"
    result = subprocess.run([str(probe)], input=text, text=True,
                            capture_output=True, check=True)
    for case, line in zip(cases, result.stdout.splitlines(), strict=True):
        growth, cover, wp, decline, stomatal = line.split()
        case["reduction_growth_pct"] = int(growth)
        case["reduction_cover_pct"] = int(cover)
        case["reduction_water_productivity_pct"] = int(wp)
        case["canopy_decline"] = float(decline)
        case["reduction_stomatal_pct"] = int(stomatal)
    output = ROOT / "tests/fixtures/fertility_fortran73.json"
    output.write_text(json.dumps(dict(
        source="Native global.f90 CropStressParametersSoilFertility and KsAny",
        precision="Unrounded native float64; percent reductions as int8 values",
        source_sha256=hashlib.sha256((REFERENCE / "global.f90").read_bytes()).hexdigest(),
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},
        cases=cases), indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    generate()
