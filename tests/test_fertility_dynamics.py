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


def run_native(probe):
    rows = native_inputs()
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


def test_dynamic_fertility_stress_matches_native_eager_jit_and_scan():
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

    def evaluate(total, unlimited, fraction, b0, b1, b2, management,
                 previous, sum_kc_top, calibrated):
        return jax.vmap(lambda total_day, unlimited_day, fraction_day, c0, c1,
                        c2, management_day, previous_day, top, calibrated_day:
            dynamic_fertility_stress_step(
                total_day, unlimited_day, fraction_day, c0, c1, c2,
                management_day, previous_day, top,
                calibrated=calibrated_day))(
                    total, unlimited, fraction, b0, b1, b2, management,
                    previous, sum_kc_top, calibrated)

    values = (jnp.asarray(biomass_total), jnp.asarray(unlimited),
              jnp.asarray(fraction), jnp.asarray(b0), jnp.asarray(b1),
              jnp.asarray(b2), jnp.asarray(management), jnp.asarray(previous),
              jnp.asarray(sum_kc_top), jnp.asarray(calibrated))
    eager = evaluate(*values)
    compiled = jax.jit(evaluate)(*values)

    scan_forcing = tuple(jnp.asarray(column) for column in
        (biomass_total, unlimited, fraction, b0, b1, b2, management,
         sum_kc_top, calibrated))

    def scan_step(previous_state, forcing):
        total, unlimited_day, fraction_day, c0, c1, c2, management_day, top, calibrated_day = forcing
        update = dynamic_fertility_stress_step(
            total, unlimited_day, fraction_day, c0, c1, c2, management_day,
            previous_state, top, calibrated=calibrated_day)
        return update.previous_stress_pct, update

    _, scanned = jax.jit(lambda x: lax.scan(
        scan_step, jnp.asarray(previous[0], jnp.int32), x))(scan_forcing)

    for result in (eager, compiled, scanned):
        np.testing.assert_array_equal(result.stress_pct, expected[:, 0])
        np.testing.assert_array_equal(result.previous_stress_pct, expected[:, 1])
        np.testing.assert_allclose(result.sum_kc_top_stress, expected[:, 4],
                                   rtol=0., atol=1e-14)
        assert result.stress_pct.dtype == jnp.int32
        assert result.sum_kc_top_stress.dtype == jnp.float64
        assert np.all(np.isfinite(result.sum_kc_top_stress))


def test_native_fertility_biomass_update_precedes_dynamic_stress():
    inputs, expected = native_inputs(), native_expected()
    row = next(index for index, case in enumerate(CASES)
               if case["name"] == "daily-biomass-is-updated-before-stress")
    _, _, biomass_before, _, _, _, _, _, _, _, eto, tact = inputs[row]
    wp = .15  # native crop WP=15 g/m2, converted to ton/ha per mm
    np.testing.assert_allclose(expected[row, 2], biomass_before + wp * tact / eto,
                               rtol=0., atol=1e-14)
    assert expected[row, 0] == 29.
