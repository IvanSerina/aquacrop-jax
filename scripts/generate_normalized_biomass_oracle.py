"""Capture Bnormalized prefixes for the weed-free calibration path."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "aquacrop-fortran/src"


def generate(output):
    base = json.loads((ROOT / "tests/fixtures/reference_et_fortran73.json").read_text())
    tail = [50, 24, 40, 52, .04, 25, 250, .1, 15., 100, 0, 0, 0, 0, 1, 0, -1.]
    cases = [dict(name=case["name"], input=case["input"][:27] + tail + [case["input"][27]])
             for case in base["cases"]]
    for name, changes in (
        ("unlimited-fertility", {27: 0, 28: 0, 29: 0, 30: 0, 31: 0., 32: 20, 33: 200}),
        ("no-wp-reduction", {30: 0}),
        ("zero-fertility-target", {27: 100}),
        ("tiny-kc-target", {43: 1e-10}),
        ("target-at-single-epsilon", {43: 2.**-22}),
        ("target-above-single-epsilon", {43: 2.**-21}),
        ("yield-switch", {36: 50, 37: 20, 38: 7, 39: 20, 40: 1}),
        ("yield-immediate", {36: 50, 39: 20, 40: 1}),
        ("salinity-wp-path", {41: 0, 42: 30}),
    ):
        for mode in (0, 1):
            row = cases[0]["input"].copy()
            row[0] = mode
            for index, value in changes.items():
                row[index] = value
            cases.append(dict(name=f"mode-{mode}-{name}", input=row))
    objects = sorted(p for p in REFERENCE.glob("*.o") if p.name != "aquacrop.o")
    text = str(len(cases)) + "\n" + "\n".join(
        " ".join(map(str, case["input"])) for case in cases) + "\n"
    with tempfile.TemporaryDirectory(prefix="normalized-biomass-") as directory:
        probe = Path(directory) / "probe"
        subprocess.run(["gfortran", "-O0", "-I", str(REFERENCE), "-o", str(probe),
            str(ROOT / "scripts/normalized_biomass_probe.f90"), *map(str, objects)],
            check=True, capture_output=True, text=True)
        result = subprocess.run([str(probe)], input=text, text=True,
                                capture_output=True, check=True)
    values = iter(map(float, result.stdout.split()))
    for case in cases:
        case["sum_kc_top"] = next(values)
        case["cumulative_biomass"] = [next(values) for _ in range(case["input"][3])]
    assert next(values, None) is None
    data = dict(source="Native tempprocessing.f90::Bnormalized, WeedStress=0; original routine per prefix",
        precision="Unrounded float64; g/m2 WP parameter and ETo=5; no production ton/ha conversion",
        source_sha256={name: hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest()
                       for name in ("global.f90", "tempprocessing.f90")},
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},
        input_fields=base["input_fields"][:27] + ["fertility_stress_pct", "reduction_growth_pct",
            "reduction_cover_pct", "reduction_wp_pct", "canopy_decline", "full_canopy_sf_day",
            "full_canopy_sf_gdd", "days_per_gdd", "water_productivity", "wp_yield_pct",
            "yield_formation_days", "switch_day", "flowering_day", "reproductive_crop",
            "fertility_mode", "reduction_stomatal_pct", "sum_kc_override", "senescence_exponent"], cases=cases)
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{output}: {len(cases)} native trajectories")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        default=ROOT / "tests/fixtures/normalized_biomass_fortran73.json")
    generate(parser.parse_args().output)
