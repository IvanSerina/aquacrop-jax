"""Capture the reference calendar's SP, final-day and annual-cycle guards."""
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
    for temperature in (6., 12., 13., 19.999, 20., 20.001, 5.95, 5.99):
        for target in (0, 1, 5, 10, 47, 461):
            cases.append(dict(name=f"constant-{temperature}-target-{target}",
                input=[target, 0, 0, temperature, temperature, 30.]))
    for target in (1, 5, 10, 500):
        for offset in (0, 363, 364):
            cases.append(dict(name=f"alternating-offset-{offset}-target-{target}",
                input=[target, offset, 1, 6., 10., 30.]))
    for target in (2, 10):
        cases.append(dict(name=f"sparse-heat-skip-last-target-{target}",
            input=[target, 0, 2, 5., 6., 30.]))
    cases.append(dict(name="negative-target", input=[-1, 0, 0, 5., 5., 30.]))
    objects = sorted(p for p in REFERENCE.glob("*.o") if p.name != "aquacrop.o")
    inputs = str(len(cases)) + "\n" + "\n".join(
        " ".join(map(str, case["input"])) for case in cases) + "\n"
    with tempfile.TemporaryDirectory(prefix="reference-calendar-") as directory:
        probe = Path(directory) / "probe"
        subprocess.run(["gfortran", "-O0", "-I", str(REFERENCE), "-o", str(probe),
            str(ROOT / "scripts/reference_calendar_probe.f90"), *map(str, objects)],
            check=True, capture_output=True, text=True)
        result = subprocess.run([str(probe)], input=inputs, capture_output=True,
                                text=True, check=True)
    for case, days in zip(cases, map(int, result.stdout.split()), strict=True):
        case["days"] = days
    data = dict(source="Native global.f90::SumCalendarDaysReferenceTnx; SP array branch",
        source_sha256=hashlib.sha256((REFERENCE / "global.f90").read_bytes()).hexdigest(),
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},
        input_fields=["target_gdd", "start_offset", "pattern", "first_temperature",
                      "second_temperature", "last_temperature"], cases=cases)
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{output}: {len(cases)} native cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        default=ROOT / "tests/fixtures/reference_calendar_fortran73.json")
    generate(parser.parse_args().output)
