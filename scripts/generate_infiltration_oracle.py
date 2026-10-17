"""Native daily no-bund infiltration fixtures; immutable Fortran objects only."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "aquacrop-fortran/src"


def generate():
    probe = ROOT / "build/infiltration-probe"
    probe.parent.mkdir(exist_ok=True)
    objects = sorted(p for p in REFERENCE.glob("*.o") if p.name != "aquacrop.o")
    if not objects:
        raise RuntimeError("Prebuilt Fortran objects required")
    subprocess.run(["gcc", "-O2", "-no-pie", "-o", str(probe),
        str(ROOT / "scripts/infiltration_probe.c"), *map(str, objects), "-l:libgfortran.so.5", "-lm"], check=True)
    inputs = [
        ("dry-fill", [.2, .2], [.1, .1], [.29, .29], [0., 0.], 1., 1200., 0., 30.),
        ("wet-top", [.4, .2], [.1, .1], [.29, .29], [0., 0.], 1., 1200., 0., 3.2),
        ("saturated", [.46, .46], [.1, .1], [.29, .29], [0., 0.], 1., 1200., 0., 8.),
        ("surface-ksat", [.2, .2], [.1, .1], [.29, .29], [0., 0.], 1., 1., 0., 30.),
        ("existing-flux", [.35, .35], [.1, .1], [.29, .29], [.8, .8], 1., 1., 0., 2.),
        ("adjusted-fc", [.43, .40, .35], [.1, .15, .2], [.43, .43, .45], [0.] * 3, 1., 1200., 0., 15.),
        ("slow-drainage", [.4, .2], [.1, .2], [.29, .29], [0., 0.], .1, 100., 0., 8.),
        ("zero-tau", [.45, .2], [.1, .1], [.29, .29], [0., 0.], 0., 1200., 0., 30.),
        ("gravel", [.4, .2], [.1, .15], [.29, .29], [0., 0.], 1., 1200., 20., 5.),
        ("no-water", [.35, .2], [.1, .1], [.29, .29], [.5, .5], 1., 1200., 0., 0.),
        ("upward-rejection", [.46] * 3, [.1, .15, .2], [.29] * 3, [.2] * 3, .2, .3, 0., 20.),
    ]
    cases = []
    for name, theta, dz, adjusted, flux, tau, ksat, gravel, water in inputs:
        text = f"{len(theta)} 46 29 {tau:.17g} {ksat:.17g} {gravel:.17g} {water:.17g}\n"
        text += "\n".join(f"{d:.17g} {t:.17g} {a * 100.:.17g} {f:.17g}"
                          for d, t, a, f in zip(dz, theta, adjusted, flux)) + "\n"
        result = subprocess.run([str(probe)], input=text, text=True, capture_output=True, check=True)
        data = [list(map(float, line.split())) for line in result.stdout.splitlines()]
        cases.append(dict(name=name, initial_theta=theta, thickness=dz, adjusted_fc=adjusted,
            initial_fluxout=flux, saturation_pct=46., field_capacity_pct=29., tau=tau, ksat=ksat,
            gravel_volume_pct=gravel, water_mm=water, theta=[row[0] for row in data[:-1]],
            fluxout=[row[1] for row in data[:-1]], deep_percolation=data[-1][0],
            rejected=data[-1][1], accepted=data[-1][2], residual=data[-1][3]))
    output = ROOT / "tests/fixtures/infiltration_fortran73.json"
    output.write_text(json.dumps(dict(
        source="AquaCrop 7.3 calculate_Extra_runoff then calculate_infiltration; daily, no bunds",
        source_sha256=hashlib.sha256((REFERENCE / "simul.f90").read_bytes()).hexdigest(),
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},
        cases=cases), indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    generate()
