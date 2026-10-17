"""Capture unrounded native potential canopy boundaries from immutable objects."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "aquacrop-fortran/src"


def probe_input(case):
    fields = [case[k] for k in ("mode", "delayed_days")]
    fields += [len(case["times"])]
    fields += [case[k] for k in ("emergence", "senescence", "maturity",
        "initial_cover", "maximum_cover", "growth", "decline",
        "reduction_growth_pct", "reduction_cover_pct")]
    return " ".join(map(str, fields)) + "\n" + " ".join(map(str, case["times"])) + "\n"


def generate():
    probe = ROOT / "build/canopy-probe"
    probe.parent.mkdir(exist_ok=True)
    objects = sorted(p for p in REFERENCE.glob("*.o") if p.name != "aquacrop.o")
    if not objects:
        raise RuntimeError("Prebuilt Fortran reference objects required")
    subprocess.run(["gcc", "-O2", "-no-pie", "-o", str(probe),
        str(ROOT / "scripts/canopy_probe.c"), *map(str, objects),
        "-l:libgfortran.so.5", "-lm"], check=True)
    cases = []
    for mode in (0, 1):
        for name, initial, maximum, growth, red_growth, red_cover in (
            ("ordinary", .01, .9, .04, 0, 0),
            ("fertility", .01, .9, .04, 35, 45),
            ("tiny-maximum", .00001, .0009, .04, 0, 0),
            ("tiny-adjusted-maximum", .000001, .09, .04, 0, 99),
            ("tiny-initial", 1e-18, .9, .04, 0, 0),
            ("zero-initial", 0., .9, .04, 0, 0),
            ("full-cover-reduction", .01, .9, .04, 0, 100),
            ("threshold-exact", .001, .001, .5, 0, 0),
            ("threshold-below-single", .001, .00100000002, .5, 0, 0),
            ("threshold-above-single", .001, .00100000006, .5, 0, 0),
            ("sub-epsilon-output", 3e-16, 1e-16, 0., 0, 0),
        ):
            for delayed in (0, 3):
                times = [-1, 0, 1, 5, 6, 50, 99, 100, 101, 119, 120, 121, 123, 124]
                if mode == 0:
                    times += [119.5, 120.499999, 120.5, 120.500001, 121.5]
                case = dict(name=f"{name}-mode{mode}-delay{delayed}", mode=mode,
                    delayed_days=delayed, emergence=5, senescence=100, maturity=120,
                    initial_cover=initial, maximum_cover=maximum, growth=growth,
                    decline=.006, reduction_growth_pct=red_growth,
                    reduction_cover_pct=red_cover, times=sorted(times))
                result = subprocess.run([str(probe)], input=probe_input(case),
                    text=True, capture_output=True, check=True)
                case["cover"] = [float(line.split()[1]) for line in result.stdout.splitlines()]
                cases.append(case)
    output = ROOT / "tests/fixtures/canopy_fortran73.json"
    output.write_text(json.dumps(dict(
        source="Native CanopyCoverNoStressSF; GDD mode 0, calendar mode 1",
        precision="Unrounded float64; cover fraction, growth/decline per day or GDD",
        source_sha256=hashlib.sha256((REFERENCE / "global.f90").read_bytes()).hexdigest(),
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},
        cases=cases), indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    generate()
