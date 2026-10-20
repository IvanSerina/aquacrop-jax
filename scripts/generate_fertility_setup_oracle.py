"""Capture real Ottawa fertility setup without writing in the reference tree."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "aquacrop-fortran"


def generate(output, testcase_root=None):
    testcase_root = REFERENCE / "testcase" if testcase_root is None else Path(testcase_root)
    objects = sorted(p for p in (REFERENCE / "src").glob("*.o")
                     if p.name != "aquacrop.o")
    if not shutil.which("gfortran") or not objects:
        raise RuntimeError("gfortran and prebuilt native reference objects required")
    cases = [dict(stress=stress, calibrated=True) for stress in range(-1, 101)]
    cases.extend(dict(stress=stress, calibrated=False) for stress in (0, 50, 100))
    with tempfile.TemporaryDirectory(prefix="fertility-setup-") as directory:
        temporary = Path(directory)
        probe = temporary / "probe"
        subprocess.run(["gfortran", "-O0", "-I", str(REFERENCE / "src"),
            "-o", str(probe), str(ROOT / "scripts/fertility_setup_probe.f90"),
            *map(str, objects)], check=True, capture_output=True, text=True)
        testcase = temporary / "testcase"
        shutil.copytree(testcase_root, testcase,
                        ignore=shutil.ignore_patterns("OUTP", "OUTP_REF"))
        (testcase / "OUTP").mkdir()
        inputs = str(len(cases)) + "\n" + "\n".join(
            f'{case["stress"]} {int(case["calibrated"])}' for case in cases) + "\n"
        result = subprocess.run([str(probe)], cwd=testcase, input=inputs,
                                text=True, capture_output=True, check=True)
        def temperatures(path, header=0):
            return [list(map(float, line.split())) for line in path.read_text().splitlines()[header:]
                    if line.strip()]
        monthly = temperatures(testcase / "SIMUL/OttawaReference.Tnx", header=8)
        daily = temperatures(testcase / "SIMUL/TnxReference365Days.SIM")
        crop_daily = temperatures(testcase / "SIMUL/TCropReference.SIM")
    lines = result.stdout.splitlines()
    assert lines[0].startswith("setup ") and lines[1].startswith("timing ")
    setup = list(map(float, lines[0].split()[1:]))
    timing = list(map(float, lines[1].split()[1:]))
    assert lines[2].startswith("samples ")
    samples = list(map(float, lines[2].split()[1:]))
    assert lines[3].startswith("reference ")
    _, reference_year, reference_co2 = lines[3].split()
    assert lines[4].startswith("thermal ") and lines[5].startswith("refcalendar ")
    thermal = list(map(float, lines[4].split()[1:]))
    reference_calendar = list(map(float, lines[5].split()[1:]))
    assert lines[6].startswith("refkc ")
    reference_kc = float(lines[6].split()[1])
    senescence_exponent = float(lines[6].split()[2])
    levels = [list(map(float, line.split()[1:])) for line in lines[7:15]]
    assert all(line.startswith("level ") for line in lines[7:15])
    for case, line in zip(cases, lines[15:], strict=True):
        stress, *values = map(float, line.split())
        assert stress == case["stress"]
        case["coefficients"] = values[:3]
        case["fraction"] = values[3]
    inputs = ["LIST/Ottawa.PRM", "DATA/AlfOttawaGDD.CRO", "DATA/Ottawa.MAN",
              "DATA/Ottawa.Tnx", "SIMUL/MaunaLoa.CO2", "PARAM/Ottawa.PPn"]
    data = dict(
        source="Native Ottawa InitializeRunPart1 and RelationshipsForFertilityAndSaltStress",
        precision="Unrounded float64; native setup in disposable testcase copy",
        source_sha256={name: hashlib.sha256((REFERENCE / "src" / name).read_bytes()).hexdigest()
                       for name in ("run.f90", "global.f90", "preparefertilitysalinity.f90")},
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},
        inputs_sha256={name: hashlib.sha256((testcase_root / name).read_bytes()).hexdigest()
                       for name in inputs},
        setup_fields=["management_stress_pct", "coefficient0", "coefficient1", "coefficient2",
                      "potential_biomass_fraction", "reduction_growth_pct", "reduction_cover_pct",
                      "reduction_wp_pct", "full_canopy_sf_day", "full_canopy_sf_gdd", "canopy_decline"],
        setup=setup,
        timing_fields=["initial_cover", "growth_coefficient", "maximum_cover", "emergence_day",
                       "full_canopy_day", "senescence_day", "flowering_day", "flowering_length",
                       "determinant_crop"],
        timing_inputs=timing, biomass_percentages=samples[3:],
        fitted_coefficients=samples[:3], reference_year=int(reference_year),
        reference_co2=float(reference_co2), reference_monthly_temperature=monthly,
        reference_daily_temperature=daily, crop_reference_temperature=crop_daily,
        thermal_fields=["emergence_gdd", "full_canopy_gdd", "senescence_gdd", "maturity_gdd", "hi_gdd",
            "thermal_growth", "thermal_decline", "calendar_growth", "calendar_decline",
            "water_productivity", "kc_top", "kc_decline_pct", "late_evaporation_pct",
            "base_temperature", "upper_temperature", "minimum_gdd", "default_tmin", "default_tmax",
            "simulation_co2", "sum_kc_top", "sum_kc_top_stress"],
        thermal_parameters=thermal,
        reference_calendar_fields=["emergence_day", "full_canopy_day", "flowering_day", "flowering_length",
            "senescence_day", "maturity_day", "hi_days", "calendar_growth", "calendar_decline", "hi_rate"],
        reference_calendar=reference_calendar, reference_sum_kc_top=reference_kc,
        senescence_exponent=senescence_exponent,
        reference_level_fields=["stress_pct", "full_canopy_sf_day", "full_canopy_sf_gdd",
            "reduction_growth_pct", "reduction_cover_pct", "reduction_wp_pct", "canopy_decline",
            "days_per_gdd", "normalized_biomass"], reference_levels=levels, cases=cases)
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{output}: {len(cases)} native cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        default=ROOT / "tests/fixtures/fertility_setup_fortran73.json")
    parser.add_argument("--testcase", type=Path, help="Input testcase to copy; never modified")
    arguments = parser.parse_args()
    generate(arguments.output, arguments.testcase)
