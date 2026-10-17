"""Generate unrounded potential ET fixtures from native CalculateETpot."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "aquacrop-fortran/src"
INPUT_ORDER = ("day_after_planting", "emergence_day", "full_canopy_day",
    "senescence_day", "harvest_day", "last_cut_day", "delayed_days",
    "evaporation_limit", "canopy_cover", "eto_mm", "maximum_crop_coefficient",
    "cumulative_crop_coefficient_decline_pct", "maximum_canopy_cover",
    "maximum_withered_canopy", "late_evaporation_effect_pct", "co2_ppm",
    "daily_gdd", "minimum_gdd_full_transpiration", "wet_bare_coefficient",
    "senescence_exponent")


def generate():
    probe = ROOT / "build/et-probe"
    probe.parent.mkdir(exist_ok=True)
    objects = sorted(p for p in REFERENCE.glob("*.o") if p.name != "aquacrop.o")
    if not objects:
        raise RuntimeError("Prebuilt Fortran reference objects required")
    subprocess.run(["gcc", "-O2", "-no-pie", "-o", str(probe),
        str(ROOT / "scripts/et_probe.c"), *map(str, objects),
        "-l:libgfortran.so.5", "-lm"], check=True)
    baseline = dict(zip(INPUT_ORDER, (40, 5, 30, 90, 120, 0, 0, 1,
        .8, 5., 1.15, 10., .9, .9, 60., 369.41, 10., 8., 1.10, 2.5)))
    cases = []
    variants = [("ordinary", {}), ("late-season", dict(day_after_planting=100)),
        ("dying", dict(day_after_planting=100, canopy_cover=.2)),
        ("beyond-maximum", dict(day_after_planting=100, canopy_cover=.95)),
        ("cut-age", dict(day_after_planting=100, last_cut_day=80)),
        ("co2", dict(co2_ppm=550.)),
        ("cold", dict(daily_gdd=3.)),
        ("negative-gdd", dict(daily_gdd=-1.)),
        ("tiny-canopy-cold", dict(canopy_cover=5e-7, daily_gdd=0.)),
        ("above-tiny-canopy", dict(canopy_cover=1e-6, daily_gdd=0.)),
        ("sub-epsilon-canopy", dict(canopy_cover=1e-17, daily_gdd=10.)),
        ("bare", dict(canopy_cover=0.)),
        ("before-emergence", dict(day_after_planting=1, canopy_cover=0.)),
        ("before-half-percent", dict(day_after_planting=1, canopy_cover=.005)),
        ("before-above-half-percent", dict(day_after_planting=1, canopy_cover=.00500000001)),
        ("after-harvest", dict(day_after_planting=121)),
        ("delayed-before-emergence", dict(day_after_planting=7, delayed_days=3, canopy_cover=.005)),
        ("delayed-harvest", dict(day_after_planting=123, delayed_days=3)),
        ("alternate-bare", dict(wet_bare_coefficient=1.2)),
        ("alternate-senescence", dict(senescence_exponent=1.)),
        ("zero-eto", dict(eto_mm=0.)),
        ("undefined-temperature", dict(minimum_gdd_full_transpiration=-9., daily_gdd=0.)),
        ("rounded-defined-temperature", dict(minimum_gdd_full_transpiration=-8.5, daily_gdd=0.)),
        ("near-coincident-temperature", dict(minimum_gdd_full_transpiration=1e-8, daily_gdd=0.)),
        ("epsilon-temperature", dict(minimum_gdd_full_transpiration=1e-7, daily_gdd=0.)),
        ("above-epsilon-temperature", dict(minimum_gdd_full_transpiration=1.00000002e-7, daily_gdd=0.)),
    ]
    for limit in (0, 1):
        for name, updates in variants:
            inputs = dict(baseline, evaporation_limit=limit, **updates)
            cases.append(dict(name=f"{name}-limit{limit}", inputs=inputs))
    text = str(len(cases)) + "\n" + "\n".join(
        " ".join(str(c["inputs"][key]) for key in INPUT_ORDER) for c in cases) + "\n"
    native = subprocess.run([str(probe)], input=text, text=True, capture_output=True, check=True)
    for case, line in zip(cases, native.stdout.splitlines(), strict=True):
        case["transpiration_mm"], case["evaporation_mm"] = map(float, line.split())
    output = ROOT / "tests/fixtures/et_fortran73.json"
    output.write_text(json.dumps(dict(source="Native CalculateETpot, explicit simulation settings",
        precision="Unrounded float64, ET in mm/day; calendar arguments int32",
        input_order=INPUT_ORDER,
        source_sha256=hashlib.sha256((REFERENCE / "global.f90").read_bytes()).hexdigest(),
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},
        cases=cases), indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    generate()
