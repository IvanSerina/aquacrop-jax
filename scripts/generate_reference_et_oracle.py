"""Native SeasonalSumOfKcPot prefixes with seeded and regrowing crops."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "aquacrop-fortran/src"


def generate(output):
    objects = sorted(p for p in REFERENCE.glob("*.o") if p.name != "aquacrop.o")
    cases = []
    for mode in (0, 1):
        for initial_day, initial_gdd in ((0, 0), (5, 40), (-9, -9)):
            for name, tmin, tmax in (("warm", 10., 22.), ("cool", 2., 10.), ("zero-gdd", -2., 8.)):
                row = [mode, initial_day, initial_gdd, 80, 3, 20, 50, 65,
                       25, 200, 500, 650, .05, .95, .18, .012, .036, .006,
                       1.15, 11., 60., 5., 30., tmin, tmax, 8., 369.41, 2.5]
                cases.append(dict(name=f"mode-{mode}-initial-{initial_day}-{name}", input=row))
        row = cases[-9]["input"].copy()
        row[-2] = 550.
        cases.append(dict(name=f"mode-{mode}-co2-550", input=row))
    for mode, index in ((0, 0), (1, 10)):
        row = cases[index]["input"].copy()
        row[-1] = 1.
        cases.append(dict(name=f"mode-{mode}-native-default-exponent", input=row))
    text = str(len(cases)) + "\n" + "\n".join(
        " ".join(map(str, case["input"])) for case in cases) + "\n"
    with tempfile.TemporaryDirectory(prefix="seasonal-kc-") as directory:
        probe = Path(directory) / "probe"
        subprocess.run(["gfortran", "-O0", "-I", str(REFERENCE), "-o", str(probe),
            str(ROOT / "scripts/seasonal_kc_probe.f90"), *map(str, objects)],
            check=True, capture_output=True, text=True)
        result = subprocess.run([str(probe)], input=text, text=True,
                                capture_output=True, check=True)
    values = iter(map(float, result.stdout.split()))
    for case in cases:
        case["cumulative_kc"] = [next(values) for _ in range(case["input"][3])]
    assert next(values, None) is None
    data = dict(source="Native global.f90::SeasonalSumOfKcPot; each daily prefix calls the original routine",
        precision="Unrounded float64; constant daily climate, ETo=5, GDD method 3",
        source_sha256=hashlib.sha256((REFERENCE / "global.f90").read_bytes()).hexdigest(),
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},
        input_fields=["mode", "initial_cover_day", "initial_cover_gdd", "days",
            "emergence_day", "full_canopy_day", "senescence_day", "maturity_day",
            "emergence_gdd", "full_canopy_gdd", "senescence_gdd", "maturity_gdd",
            "initial_cover", "maximum_cover", "calendar_growth", "thermal_growth",
            "calendar_decline", "thermal_decline", "kc_top", "kc_decline_pct",
            "late_evaporation_pct", "base_temperature", "upper_temperature",
            "tmin", "tmax", "minimum_gdd", "co2_ppm", "senescence_exponent"], cases=cases)
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{output}: {len(cases)} native trajectories")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        default=ROOT / "tests/fixtures/reference_et_fortran73.json")
    generate(parser.parse_args().output)
