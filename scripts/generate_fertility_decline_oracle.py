"""Capture CCiNoWaterStressSF at calendar and thermal guard boundaries."""
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
    for mode in (0, 1):
        for decline in (0., 1e-6, 2e-6, .04, 30.):
            for day in (0, 3, 22, 23, 49, 50, 51, 65, 80, 120):
                row = [mode, day, 3, 22, 50, 65, 25, 220, 500, 650,
                       .05, .95, .18, .012, .036, .006, day * 10., .1, 24, 40, decline]
                cases.append(dict(name=f"mode-{mode}-decline-{decline}-day-{day}", input=row))
    for name, day, gdd, full_gdd, sen_gdd, full_day, cdc, gddcdc in (
        ("thermal-before-full", 30, 219., 220, 500, 22, .036, .006),
        ("thermal-at-full", 30, 220., 220, 500, 22, .036, .006),
        ("thermal-after-full", 30, 221., 220, 500, 22, .036, .006),
        ("calendar-before-full-thermal-late", 22, 550., 220, 500, 22, .036, .006),
        ("calendar-late-thermal-early", 50, 210., 220, 500, 22, .036, .006),
        ("thermal-full-after-senescence", 60, 610., 600, 500, 22, .036, .006),
        ("calendar-full-at-senescence", 60, 610., 220, 500, 50, .036, .006),
        ("zero-decline-coefficient", 60, 600., 220, 500, 22, 0., 0.),
    ):
        for mode in (0, 1):
            cases.append(dict(name=f"mode-{mode}-{name}", input=[mode, day, 3,
                full_day, 50, 65, 25, full_gdd, sen_gdd, 650, .05, .95,
                .18, .012, cdc, gddcdc, gdd, .1, 24, 40, .04]))
    objects = sorted(p for p in REFERENCE.glob("*.o") if p.name != "aquacrop.o")
    inputs = str(len(cases)) + "\n" + "\n".join(
        " ".join(map(str, case["input"])) for case in cases) + "\n"
    with tempfile.TemporaryDirectory(prefix="fertility-decline-") as directory:
        probe = Path(directory) / "probe"
        subprocess.run(["gfortran", "-O0", "-I", str(REFERENCE), "-o", str(probe),
            str(ROOT / "scripts/fertility_decline_probe.f90"), *map(str, objects)],
            check=True, capture_output=True, text=True)
        result = subprocess.run([str(probe)], input=inputs, text=True,
                                capture_output=True, check=True)
    for case, value in zip(cases, map(float, result.stdout.split()), strict=True):
        case["cover"] = value
    data = dict(source="Native global.f90::CCiNoWaterStressSF; delayed germination=0",
        source_sha256=hashlib.sha256((REFERENCE / "global.f90").read_bytes()).hexdigest(),
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},
        input_fields=["mode", "day", "emergence_day", "full_canopy_sf_day", "senescence_day",
            "maturity_day", "emergence_gdd", "full_canopy_sf_gdd", "senescence_gdd", "maturity_gdd",
            "initial_cover", "maximum_cover", "calendar_growth", "thermal_growth", "calendar_decline",
            "thermal_decline", "cumulative_gdd", "days_per_gdd", "reduction_growth_pct",
            "reduction_cover_pct", "canopy_decline"], cases=cases)
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{output}: {len(cases)} native cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        default=ROOT / "tests/fixtures/fertility_decline_fortran73.json")
    generate(parser.parse_args().output)
