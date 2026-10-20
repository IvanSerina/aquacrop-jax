"""Native parity for the daily fertility stress update in simul.f90."""
from ctypes.util import find_library
import json
from pathlib import Path
import shutil
import subprocess

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pytest

from aquacrop_jax.production import dynamic_fertility_stress_step


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads(
    (ROOT / "tests/fixtures/fertility_dynamics_fortran73.json").read_text())
CASES = FIXTURE["cases"]
CROP_DEFAULTS = dict(grain_crop=False, determinacy_linked=False, dayi=1,
                     delayed_days=0, crop_day1=100, flowering_day=100,
                     flowering_length=0)


def crop_inputs(case):
    return CROP_DEFAULTS | case.get("crop", {})


@pytest.fixture(scope="module")
def fertility_stress_probe(tmp_path_factory):
    reference = ROOT.parent / "aquacrop-fortran/src"
    objects = sorted(p for p in reference.glob("*.o") if p.name != "aquacrop.o")
    compiler = shutil.which("gfortran")
    if not compiler or not objects or not find_library("gfortran"):
        pytest.skip("Optional native probe requires gfortran and prebuilt reference objects")
    probe = tmp_path_factory.mktemp("fertility-stress-probe") / "probe"
    result = subprocess.run(
        [compiler, "-O0", "-I", str(reference), "-o", str(probe),
         str(ROOT / "scripts/fertility_stress_probe.f90"),
         *map(str, objects)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return probe


def native_inputs():
    return np.asarray([case["input"] for case in CASES], dtype=np.float64)


def native_expected():
    return np.asarray([case["expected"] for case in CASES], dtype=np.float64)


def run_native(probe, cases=None):
    cases = CASES if cases is None else cases
    rows = [case["input"] + list(crop_inputs(case).values()) for case in cases]
    text = str(len(rows)) + "\n" + "\n".join(
        " ".join(format(value, ".17g") for value in row) for row in rows) + "\n"
    result = subprocess.run([str(probe)], input=text, text=True,
                            capture_output=True, check=True)
    return np.asarray([[float(value) for value in line.split()]
                       for line in result.stdout.splitlines()])


def test_fertility_dynamics_fixture_matches_live_native_routine(fertility_stress_probe):
    actual = run_native(fertility_stress_probe)
    expected = native_expected()
    np.testing.assert_array_equal(actual[:, :2], expected[:, :2])
    np.testing.assert_allclose(actual[:, 2:], expected[:, 2:], rtol=0., atol=1e-14)


def test_dynamic_fertility_stress_matches_native_eager_and_jit():
    inputs = native_inputs()
    expected = native_expected()
    # The native routine updates BiomassTot before calculating BioAdj. Feed
    # that post-production state into the pure JAX stress operator.
    biomass_total = expected[:, 2]
    unlimited, fraction, b0, b1, b2 = (inputs[:, index] for index in (3, 4, 5, 6, 7))
    management = inputs[:, 0].astype(np.int32)
    calibrated = inputs[:, 1].astype(bool)
    previous = inputs[:, 8].astype(np.int32)
    sum_kc_top = inputs[:, 9]
    crop_columns = tuple(jnp.asarray([crop_inputs(case)[key] for case in CASES])
                         for key in CROP_DEFAULTS)

    def evaluate(total, unlimited, fraction, b0, b1, b2, management,
                 previous, sum_kc_top, calibrated, *crop_columns):
        return jax.vmap(lambda total_day, unlimited_day, fraction_day, c0, c1,
                        c2, management_day, previous_day, top, calibrated_day,
                        grain, determinate, day, delayed, day1, flowering, length:
            dynamic_fertility_stress_step(
                total_day, unlimited_day, fraction_day, c0, c1, c2,
                management_day, previous_day, top,
                calibrated=calibrated_day, grain_crop=grain,
                determinacy_linked=determinate, dayi=day, delayed_days=delayed,
                crop_day1=day1, flowering_day=flowering,
                flowering_length=length))(
                    total, unlimited, fraction, b0, b1, b2, management,
                    previous, sum_kc_top, calibrated, *crop_columns)

    values = (jnp.asarray(biomass_total), jnp.asarray(unlimited),
              jnp.asarray(fraction), jnp.asarray(b0), jnp.asarray(b1),
              jnp.asarray(b2), jnp.asarray(management), jnp.asarray(previous),
              jnp.asarray(sum_kc_top), jnp.asarray(calibrated), *crop_columns)
    eager = evaluate(*values)
    compiled = jax.jit(evaluate)(*values)

    for result in (eager, compiled):
        np.testing.assert_array_equal(result.stress_pct, expected[:, 0])
        np.testing.assert_array_equal(result.previous_stress_pct, expected[:, 1])
        np.testing.assert_allclose(result.sum_kc_top_stress, expected[:, 4],
                                   rtol=0., atol=1e-14)
        assert result.stress_pct.dtype == jnp.int32
        assert result.sum_kc_top_stress.dtype == jnp.float64
        assert np.all(np.isfinite(result.sum_kc_top_stress))

    # Retain scan coverage for the original independent forage cases, whose
    # stress update does not depend on the carried previous stress.
    forage_rows = [i for i, case in enumerate(CASES)
                   if not crop_inputs(case)["grain_crop"]]
    scan_forcing = tuple(column[jnp.asarray(forage_rows)] for column in
                         (values[:7] + values[8:10]))

    def forage_step(previous_state, forcing):
        total, unlimited, fraction, c0, c1, c2, management, top, calibrated = forcing
        update = dynamic_fertility_stress_step(
            total, unlimited, fraction, c0, c1, c2, management,
            previous_state, top, calibrated=calibrated)
        return update.previous_stress_pct, update

    _, scanned = jax.jit(lambda forcing: lax.scan(
        forage_step, jnp.int32(previous[0]), forcing))(scan_forcing)
    np.testing.assert_array_equal(scanned.stress_pct, expected[forage_rows, 0])
    np.testing.assert_array_equal(scanned.previous_stress_pct, expected[forage_rows, 1])
    np.testing.assert_allclose(scanned.sum_kc_top_stress, expected[forage_rows, 4],
                               rtol=0., atol=1e-14)


def test_grain_stress_scan_carries_native_previous_stress(fertility_stress_probe):
    days = [102, 109, 110, 111, 112, 113, 114]
    coefficients = [.4, .2, .2, .2, .6, .2, .1]
    previous = 40
    native = []
    for day, coefficient in zip(days, coefficients):
        case = dict(input=[50, 1, .5, 1., .5, 0., coefficient, 0.,
                           previous, 100., 0., 0.],
                    crop=dict(grain_crop=True, determinacy_linked=True,
                              dayi=day, crop_day1=100, flowering_day=10,
                              flowering_length=4))
        output = run_native(fertility_stress_probe, [case])[0]
        previous = int(output[1])
        native.append(output)
    native = np.asarray(native)
    np.testing.assert_array_equal(native[:, 0], [40, 40, 40, 10, 30, 30, 30])

    def step(previous, forcing):
        day, coefficient = forcing
        update = dynamic_fertility_stress_step(
            .5, 1., .5, 0., coefficient, 0., 50, previous, 100.,
            grain_crop=True, determinacy_linked=True, dayi=day,
            crop_day1=100, flowering_day=10, flowering_length=4)
        return update.previous_stress_pct, update

    def evaluate(forcing):
        return lax.scan(step, jnp.int32(40), forcing)

    forcing = jnp.asarray(days, jnp.int32), jnp.asarray(coefficients, jnp.float64)
    for final, actual in (evaluate(forcing), jax.jit(evaluate)(forcing)):
        np.testing.assert_array_equal(actual.stress_pct, native[:, 0])
        np.testing.assert_array_equal(actual.previous_stress_pct, native[:, 1])
        np.testing.assert_allclose(actual.sum_kc_top_stress, native[:, 4],
                                   rtol=0., atol=1e-14)
        assert int(final) == int(native[-1, 1])


def test_native_fertility_biomass_update_precedes_dynamic_stress():
    inputs, expected = native_inputs(), native_expected()
    row = next(index for index, case in enumerate(CASES)
               if case["name"] == "daily-biomass-is-updated-before-stress")
    _, _, biomass_before, _, _, _, _, _, _, _, eto, tact = inputs[row]
    wp = .15  # native crop WP=15 g/m2, converted to ton/ha per mm
    np.testing.assert_allclose(expected[row, 2], biomass_before + wp * tact / eto,
                               rtol=0., atol=1e-14)
    assert expected[row, 0] == 29.
