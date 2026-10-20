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


def generate(output):
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
        shutil.copytree(REFERENCE / "testcase", testcase,
                        ignore=shutil.ignore_patterns("OUTP", "OUTP_REF"))
        (testcase / "OUTP").mkdir()
        inputs = str(len(cases)) + "\n" + "\n".join(
            f'{case["stress"]} {int(case["calibrated"])}' for case in cases) + "\n"
        result = subprocess.run([str(probe)], cwd=testcase, input=inputs,
                                text=True, capture_output=True, check=True)
    lines = result.stdout.splitlines()
    assert lines[0].startswith("setup ") and lines[1].startswith("timing ")
    setup = list(map(float, lines[0].split()[1:]))
    timing = list(map(float, lines[1].split()[1:]))
    assert lines[2].startswith("samples ")
    samples = list(map(float, lines[2].split()[1:]))
    for case, line in zip(cases, lines[3:], strict=True):
        stress, *values = map(float, line.split())
        assert stress == case["stress"]
        case["coefficients"] = values[:3]
        case["fraction"] = values[3]
    inputs = ["LIST/Ottawa.PRM", "DATA/AlfOttawaGDD.CRO", "DATA/Ottawa.MAN",
              "DATA/Ottawa.Tnx"]
    data = dict(
        source="Native Ottawa InitializeRunPart1 and RelationshipsForFertilityAndSaltStress",
        precision="Unrounded float64; native setup in disposable testcase copy",
        source_sha256={name: hashlib.sha256((REFERENCE / "src" / name).read_bytes()).hexdigest()
                       for name in ("run.f90", "global.f90", "preparefertilitysalinity.f90")},
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},
        inputs_sha256={name: hashlib.sha256((REFERENCE / "testcase" / name).read_bytes()).hexdigest()
                       for name in inputs},
        setup_fields=["management_stress_pct", "coefficient0", "coefficient1", "coefficient2",
                      "potential_biomass_fraction", "reduction_growth_pct", "reduction_cover_pct",
                      "reduction_wp_pct", "full_canopy_sf_day", "full_canopy_sf_gdd", "canopy_decline"],
        setup=setup,
        timing_fields=["initial_cover", "growth_coefficient", "maximum_cover", "emergence_day",
                       "full_canopy_day", "senescence_day", "flowering_day", "flowering_length",
                       "determinant_crop"],
        timing_inputs=timing, biomass_percentages=samples[3:],
        fitted_coefficients=samples[:3], cases=cases)
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{output}: {len(cases)} native cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        default=ROOT / "tests/fixtures/fertility_setup_fortran73.json")
    generate(parser.parse_args().output)
