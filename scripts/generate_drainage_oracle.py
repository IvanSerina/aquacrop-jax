"""Capture unrounded calculate_drainage results without modifying Fortran."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "aquacrop-fortran/src"


def generate():
    probe = ROOT / "build/drainage-probe"
    probe.parent.mkdir(exist_ok=True)
    objects = sorted(p for p in REFERENCE.glob("*.o") if p.name != "aquacrop.o")
    if not objects:
        raise RuntimeError("Prebuilt Fortran reference objects required")
    subprocess.run(["gcc", "-O2", "-no-pie", "-o", str(probe),
        str(ROOT / "scripts/drainage_probe.c"), *map(str, objects), "-l:libgfortran.so.5", "-lm"], check=True)
    inputs = [
        ("uniform-wet", [.35, .35], [.1, .1], [.29, .29], 1., 1200., 0.),
        ("wet-over-dry", [.40, .13], [.1, .1], [.29, .29], 1., 1200., 0.),
        ("dry-over-wet", [.13, .40], [.1, .1], [.29, .29], 1., 1200., 0.),
        ("wet-over-fc", [.40, .29], [.1, .1], [.29, .29], 1., 1200., 0.),
        ("adjusted-fc", [.45, .40, .35], [.1, .15, .2], [.40, .43, .45], 1., 1200., 0.),
        ("unequal-thickness", [.40, .13, .18], [.1, .15, .2], [.29] * 3, .5, 1200., 0.),
        ("gravel", [.40, .13, .18], [.1, .15, .2], [.29] * 3, .5, 1200., 20.),
        ("ksat-limit", [.35, .35], [.1, .1], [.29, .29], 1., 1., 0.),
        ("zero-tau", [.37, .20], [.1, .1], [.29, .29], 0., 1200., 0.),
        ("saturated", [.46] * 3, [.1, .15, .2], [.29] * 3, 1., 1200., 0.),
        ("wet-dry-backflow", [.46, .13, .45, .15], [.1, .1, .15, .2], [.29] * 4, .8, 3., 0.),
        ("low-ksat-dry-bottom", [.46, .46, .13], [.1, .15, .2], [.29] * 3, 1., .2, 0.),
    ]
    cases = []
    for name, theta, dz, adjusted, tau, ksat, gravel in inputs:
        text = f"{len(theta)} 46 29 {tau:.17g} {ksat:.17g} {gravel:.17g}\n"
        text += "\n".join(f"{d:.17g} {t:.17g} {a * 100.:.17g}"
                          for d, t, a in zip(dz, theta, adjusted)) + "\n"
        result = subprocess.run([str(probe)], input=text, text=True, capture_output=True, check=True)
        data = [list(map(float, line.split())) for line in result.stdout.splitlines()]
        cases.append(dict(name=name, initial_theta=theta, thickness=dz, adjusted_fc=adjusted,
            saturation_pct=46., field_capacity_pct=29., tau=tau, ksat=ksat, gravel_volume_pct=gravel,
            theta=[row[0] for row in data[:-1]], fluxout=[row[1] for row in data[:-1]],
            drainage=data[-1][0], residual=data[-1][1]))
    fixture = dict(source="AquaCrop 7.3 native calculate_drainage, homogeneous layer; unrounded float64",
        source_sha256=hashlib.sha256((REFERENCE / "simul.f90").read_bytes()).hexdigest(),
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects}, cases=cases)
    output = ROOT / "tests/fixtures/drainage_fortran73.json"
    output.write_text(json.dumps(fixture, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    generate()
