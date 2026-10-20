"""Capture actual AdjustCalendarDaysReferenceTnx for four crop categories."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "aquacrop-fortran/src"


def generate(output):
    cases = []
    for kind in (1, 2, 3, 4):
        for initial_day, initial_gdd in ((0, 0), (5, 40), (-9, -9)):
            # At least 3 GDD/day keeps flowering offsets within the native
            # annual array bounds; colder synthetic crops can read past it.
            for temperature in (8., 16.):
                row = [kind, initial_day, initial_gdd, temperature,
                       2, 39, 180, 180, 30, 10, 5, 461, 1803, 1920, 400, 100,
                       118, 50, .05, .95, .17713, .012, .03636, .006, 17, 50. / 17.]
                cases.append(dict(name=f"kind-{kind}-initial-{initial_day}-temp-{temperature}", input=row))
    row = cases[0]["input"].copy()
    row[16] = 0
    cases.append(dict(name="no-hi-period", input=row))
    objects = sorted(p for p in REFERENCE.glob("*.o") if p.name != "aquacrop.o")
    inputs = str(len(cases)) + "\n" + "\n".join(
        " ".join(map(str, case["input"])) for case in cases) + "\n"
    with tempfile.TemporaryDirectory(prefix="reference-crop-calendar-") as directory:
        probe = Path(directory) / "probe"
        subprocess.run(["gfortran", "-O0", "-I", str(REFERENCE), "-o", str(probe),
            str(ROOT / "scripts/reference_crop_calendar_probe.f90"), *map(str, objects)],
            check=True, capture_output=True, text=True)
        result = subprocess.run([str(probe)], input=inputs, text=True,
                                capture_output=True, check=True)
    for case, line in zip(cases, result.stdout.splitlines(), strict=True):
        case["expected"] = list(map(float, line.split()))
    data = dict(source="Native preparefertilitysalinity.f90::AdjustCalendarDaysReferenceTnx",
        source_sha256={name: hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest()
                       for name in ("global.f90", "tempprocessing.f90", "preparefertilitysalinity.f90")},
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},
        input_fields=["legacy_crop_kind", "initial_cover_day", "initial_cover_gdd", "temperature",
            "emergence_day", "full_canopy_day", "senescence_day", "maturity_day", "flowering_day", "flowering_length",
            "emergence_gdd", "full_canopy_gdd", "senescence_gdd", "maturity_gdd", "flowering_gdd", "flowering_length_gdd",
            "hi_gdd", "reference_hi_pct", "initial_cover", "maximum_cover", "calendar_growth", "thermal_growth",
            "calendar_decline", "thermal_decline", "hi_days", "hi_rate"],
        output_fields=["emergence_day", "full_canopy_day", "flowering_day", "flowering_length",
            "senescence_day", "maturity_day", "hi_days", "calendar_growth", "calendar_decline", "hi_rate"], cases=cases)
    output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(f"{output}: {len(cases)} native cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        default=ROOT / "tests/fixtures/reference_crop_calendar_fortran73.json")
    generate(parser.parse_args().output)
