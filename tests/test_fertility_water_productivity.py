"""Actual daily production WP against the original linked Fortran routine."""
from ctypes.util import find_library
from pathlib import Path
import shutil
import subprocess

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pytest

from aquacrop_jax.production import fertility_water_productivity_step

ROOT = Path(__file__).resolve().parents[1]
# cumulative Kc, stressed seasonal Kc, RedWP, actual Tr, ETo
CASES = np.asarray([
    [0., 10., 0, 2., 5.], [0., 10., 50, 2., 5.],
    [9., 10., 50, 10., 5.], [10., 10., 50, 2., 5.],
    [11., 10., 50, 2., 5.], [10., 10., 0, 2., 5.],
    [0., 10., 50, 0., 5.], [0., 10., 50, 2., 0.],
    [4., 10., 0, 2., 0.], [4., 10., 50, 2., -1.],
    [0., 0., 50, 2., 5.], [1., 0., 50, 2., 5.],
    [0., 0., 0, 2., 5.], [9., 10., 100, 10., 5.],
    [np.nextafter(1., 0.), 1., 50, .5, 5.],
    [1. - np.finfo(float).eps, 1., 50, .5, 5.],
    [np.nextafter(1. - np.finfo(float).eps, 0.), 1., 50, .5, 5.],
], dtype=np.float64)


@pytest.fixture(scope="module")
def native_probe(tmp_path_factory):
    reference = ROOT.parent / "aquacrop-fortran/src"
    objects = sorted(p for p in reference.glob("*.o") if p.name != "aquacrop.o")
    if not shutil.which("gfortran") or not find_library("gfortran") or not objects:
        pytest.skip("Requires gfortran and immutable native reference objects")
    probe = tmp_path_factory.mktemp("fertility-wp") / "probe"
    built = subprocess.run(["gfortran", "-O0", "-I", str(reference), "-o", str(probe),
        str(ROOT / "scripts/fertility_stress_probe.f90"), *map(str, objects)],
        capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    return probe


def native_values(probe, rows):
    lines = [str(len(rows))]
    for cumulative, top, reduction, tact, eto in rows:
        # Forage/regrowth, CO2 reference, no weeds or assimilate transfers;
        # native WP=15/100. Other inputs preserve the existing stress probe.
        inputs = [50, 1, .5, 1., .5, 0., .4, 0., 50, 100., eto, tact,
                  0, 0, 1, 0, 100, 100, 0]
        lines.append(" ".join(format(v, ".17g") for v in inputs))
        lines.append(f"{int(reduction)} {cumulative:.17g} {top:.17g}")
    result = subprocess.run([str(probe), "--fertility-wp"],
        input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
    return np.asarray([list(map(float, line.split())) for line in result.stdout.splitlines()])


def test_wp_matches_native_thresholds_overshoot_and_no_eto(native_probe):
    expected = native_values(native_probe, CASES)
    args = (jnp.full(len(CASES), .15), *(jnp.asarray(v) for v in CASES.T))
    for operation in (fertility_water_productivity_step,
                      jax.jit(fertility_water_productivity_step)):
        actual = operation(*args)
        np.testing.assert_allclose(np.asarray(actual).T, expected, rtol=0., atol=1e-14)
        assert actual.water_productivity.dtype == jnp.float64
        # Overshoot on entry below threshold is native, even negative WP.
        assert float(actual.water_productivity[13]) < 0.
        np.testing.assert_array_equal(np.asarray(actual.sum_kci)[[3, 4]], CASES[[3, 4], 0])
        np.testing.assert_array_equal(np.asarray(actual.sum_kci)[[7, 8, 9]], CASES[[7, 8, 9], 0])
        # epsilon is DP: equality freezes; the previous representable value updates.
        assert float(actual.sum_kci[-2]) == CASES[-2, 0]
        assert float(actual.sum_kci[-1]) > CASES[-1, 0]


def test_wp_scan_carries_native_cumulative_kc(native_probe):
    top = np.asarray([1., 1., 1., 2., 2., 2., 2.])
    reduction = np.asarray([50., 50., 50., 50., 50., 0., 50.])
    tact = np.asarray([3., 3., 3., 0., 2., 3., 1.])
    eto = np.asarray([5., 5., 5., 0., 5., 5., 5.])
    cumulative, expected = 0., []
    for seasonal, red, tr, et in zip(top, reduction, tact, eto):
        row = native_values(native_probe, [[cumulative, seasonal, red, tr, et]])[0]
        cumulative = row[1]
        expected.append(row)

    def step(carry, forcing):
        update = fertility_water_productivity_step(.15, carry, *forcing)
        return update.sum_kci, update

    _, actual = jax.jit(lambda forcing: lax.scan(step, jnp.float64(0.), forcing))(
        tuple(jnp.asarray(v) for v in (top, reduction, tact, eto)))
    np.testing.assert_allclose(np.asarray(actual).T, expected, rtol=0., atol=1e-14)


def test_wp_preserves_zero_eto_and_unreduced_accumulation_without_native():
    zero = jax.jit(fertility_water_productivity_step)(.2, 4., 10., 50., 3., 0.)
    np.testing.assert_array_equal(zero, [.2, 4., 0.])
    unmodified = fertility_water_productivity_step(.2, 11., 10., 0., 3., 5.)
    np.testing.assert_array_equal(unmodified[:2], [.2, 11.6])
    np.testing.assert_allclose(unmodified.biomass_increment, .12, rtol=0., atol=1e-14)
