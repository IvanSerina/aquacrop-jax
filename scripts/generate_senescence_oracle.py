"""Capture full native DetermineCCiGDD state before natural senescence."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / "aquacrop-fortran/src"
PARAMETERS = dict(initial_cover=.05, maximum_cover=.95, growth=.012,
                  decline=.006, reduction_growth_pct=0, reduction_cover_pct=0,
                  p_senescence=.65, senescence_shape=3., fao_correction=1,
                  adjustment_factor=1., beta_pct=10, delay_eto_mm=40)
OUTPUTS = ["cover", "initial_adjusted", "maximum_adjusted", "protected",
           "time_gdd", "top_cover", "withered_cover", "eto_stress_sum_mm",
           "evaporation_limited", "no_more_crop", "leaf_stress_pct",
           "senescence_stress_pct", "p_senescence_active"]


def day(actual=36., total=100., daily=10., eto=5., topsoil=False, top_actual=None):
    return [total, daily, eto, actual, 100., 20.,
            actual if top_actual is None else top_actual, 100., 20., int(topsoil)]


def cases():
    base = dict(parameters=PARAMETERS, canopy_state=[.1, .05, .1, 0],
                senescence_state=[0., 0., .1, 0., 0, 0], forcing=[day()])
    result = []

    def add(name, *, parameters=None, canopy=None, state=None, forcing=None):
        case = deepcopy(base)
        case["name"] = name
        case["parameters"].update(parameters or {})
        if canopy is not None:
            case["canopy_state"] = canopy
        if state is not None:
            case["senescence_state"] = state
        if forcing is not None:
            case["forcing"] = forcing
        result.append(case)

    add("drought-beta-rewatering-renewed", forcing=[
        day(36., 100.), day(36., 110.), day(50., 120.),
        day(90., 130.), day(36., 140.)])
    # The native predicate is Actual < FC - pSen * (FC - WP).
    for name, water in (("wetter", math.nextafter(48., math.inf)),
                        ("equal", 48.),
                        ("drier", 48. - 1e-7),
                        ("drier-ulp", math.nextafter(48., -math.inf))):
        add("threshold-" + name, forcing=[day(water)])
    add("topsoil-wet-root-dry", forcing=[day(20., topsoil=True, top_actual=90.)])
    add("topsoil-dry-root-wet", forcing=[day(90., topsoil=True, top_actual=36.)])
    add("protected-seedling", canopy=[.05, .05, .05, 1],
        state=[0., 0., .05, 0., 0, 0], forcing=[day(20., 10., 5.)])
    add("protection-ends-before-senescence", canopy=[.061, .05, .061, 1],
        state=[0., 0., .061, 0., 0, 0], forcing=[day(20., 30.)])
    for name, time in (("equal", .0001), ("above", math.nextafter(.0001, math.inf))):
        add("beta-time-" + name, state=[time, .1, .1, 3., 1, 0], forcing=[day(50.)])
    add("fao-eto-correction", forcing=[day(36., eto=7.)])
    add("no-fao-no-beta", parameters=dict(fao_correction=0),
        state=[10., .1, .1, 3., 1, 0], forcing=[day(50., eto=7.)])
    add("fao-upper-cap", parameters=dict(p_senescence=1.), forcing=[day(20., eto=0.)])
    add("fao-lower-cap", parameters=dict(p_senescence=.1), forcing=[day(90., eto=20.)])
    add("wilting-point", forcing=[day(20.)])
    add("ks-below-zero-threshold", forcing=[day(100. - .9989999999 * 80.)])
    add("full-decline-boundary", forcing=[day(100. - .999 * 80.)])
    add("zero-daily-gdd", forcing=[day(20., daily=0.)])
    add("fertility-scaled-decline", parameters=dict(reduction_growth_pct=24,
        reduction_cover_pct=40), forcing=[day(36.), day(90., 110.)])
    add("dormancy-floor", parameters=dict(initial_cover=.01),
        canopy=[.051, .01, .051, 0], state=[10., .06, .1, 20., 1, 0],
        forcing=[day(20., 200., 100.), day(90., 210.)])
    for name, delay in (("delayed", 40), ("no-delay", 0)):
        add("small-top-" + name, parameters=dict(initial_cover=.01, delay_eto_mm=delay),
            canopy=[.0004, .0004, .0004, 0], state=[10., .0004, .1, 10., 1, 0],
            forcing=[day(20.)])
    add("overflow-guard", parameters=dict(delay_eto_mm=0),
        state=[1e6, .1, .1, 0., 1, 0], forcing=[day(20.)])
    add("log-guard", parameters=dict(delay_eto_mm=0),
        canopy=[.105, .05, .105, 0], state=[10., .1, .105, 0., 1, 0])
    add("no-more-crop-sticky", state=[0., 0., .1, 0., 0, 1], forcing=[day(90.)])
    # Water-driven senescence is also reached after the canopy has stopped
    # expanding. Exercise DetermineCCiGDD's mid-season branch with the
    # persistent full-canopy state supplied at its boundary.
    add("midseason-water-senescence", canopy=[.95, .05, .95, 0],
        state=[0., 0., .95, 0., 0, 0], forcing=[
            day(36., 700.), day(36., 710.), day(90., 720.),
            day(36., 730.), day(36., 740.)])
    result[-1]["phase"] = "midseason"
    return result


def probe_input(case):
    p = case["parameters"]
    fields = [len(case["forcing"]), *[p[k] for k in PARAMETERS]]
    return (" ".join(map(str, fields)) + "\n" +
            " ".join(map(str, case["canopy_state"] + case["senescence_state"])) + "\n" +
            "\n".join(" ".join(map(str, row)) for row in case["forcing"]) + "\n")


def generate():
    probe = ROOT / "build/senescence-probe"
    probe.parent.mkdir(exist_ok=True)
    objects = sorted(p for p in REFERENCE.glob("*.o") if p.name != "aquacrop.o")
    if not objects:
        raise RuntimeError("Prebuilt Fortran reference objects required")
    subprocess.run(["gcc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2",
        "-no-pie", "-o", str(probe), str(ROOT / "scripts/senescence_probe.c"),
        *map(str, objects), "-l:libgfortran.so.5", "-lm"], check=True)
    rows = cases()
    for case in rows:
        case["native_input"] = probe_input(case)
        result = subprocess.run([str(probe)], input=case["native_input"],
                                text=True, capture_output=True, check=True)
        native = [list(map(float, line.split())) for line in result.stdout.splitlines()]
        case["native_finite"] = all(all(map(math.isfinite, row)) for row in native)
        if not case["native_finite"]:
            assert case["name"] == "threshold-drier-ulp", case["name"]
            case["native_limitation"] = (
                "Actual is one ULP below the strict onset threshold, but KsSen rounds to 1; "
                "GDDCDCadjusted becomes 0 and the native log-time recurrence yields NaN. "
                "The JAX validity flag must reject this state without a threshold tolerance.")
        case["expected"] = [[value if math.isfinite(value) else None for value in row]
                            for row in native]
        assert len(case["expected"]) == len(case["forcing"])
        assert all(len(row) == len(OUTPUTS) for row in case["expected"])
    output = ROOT / "tests/fixtures/senescence_fortran73.json"
    output.write_text(json.dumps(dict(
        source="Native simul.f90::DetermineCCiGDD, before full canopy and natural senescence",
        precision="Unrounded float64: cover fractions, GDD, water/ETo mm, stresses percent",
        milestones=dict(emergence=5, full_canopy_sf=600, senescence=1920, maturity=2200),
        output_columns=OUTPUTS,
        source_sha256={name: hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest()
                       for name in ("simul.f90", "global.f90")},
        objects_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},
        cases=rows), indent=2) + "\n")
    print(f"{len(rows)} native cases: {output}")


if __name__ == "__main__":
    generate()
