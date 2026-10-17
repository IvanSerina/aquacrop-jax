"""Full-precision native drainage fixtures, including wet-over-dry fronts."""
import json
from pathlib import Path
from ctypes.util import find_library
import shutil
import subprocess

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pytest

from aquacrop_jax.soil import profile_water_mm, vertical_drainage

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "tests/fixtures/drainage_fortran73.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_homogeneous_drainage_matches_native_state_flux_and_balance(case):
    gravel = [case["gravel_volume_pct"]] * len(case["thickness"])
    arguments = (case["initial_theta"], case["thickness"], gravel, case["adjusted_fc"],
                 case["saturation_pct"], case["field_capacity_pct"], case["tau"], case["ksat"])
    # Existing matching branches differ from native by <= 3e-14 mm.
    # Compare water contents and every interface flux, not only total mass.
    for operation in (vertical_drainage.__wrapped__, vertical_drainage):
        theta, flux, bottom = operation(*arguments)
        assert theta.dtype == jnp.float64
        np.testing.assert_allclose(theta, case["theta"], rtol=0., atol=1e-12)
        np.testing.assert_allclose(flux, case["fluxout"], rtol=0., atol=1e-10)
        np.testing.assert_allclose(bottom, case["drainage"], rtol=0., atol=1e-10)
        before = profile_water_mm(arguments[0], arguments[1], gravel)
        after = profile_water_mm(theta, arguments[1], gravel)
        assert abs(float(before - after - bottom)) < 1e-10
        assert np.all(np.asarray(theta) >= 0.)
        assert np.all(np.asarray(theta) <= case["saturation_pct"] / 100. + 1e-12)


def test_drainage_state_can_be_carried_by_scan():
    case = next(c for c in CASES if c["name"] == "wet-over-dry")
    arguments = (case["thickness"], [0., 0.], case["adjusted_fc"], 46., 29., 1., 1200.)

    @jax.jit
    def trajectory(theta):
        def step(current, _):
            updated, flux, bottom = vertical_drainage(current, *arguments)
            return updated, (updated, flux, bottom)
        return lax.scan(step, theta, xs=None, length=3)[1]

    theta, flux, bottom = trajectory(jnp.asarray(case["initial_theta"], jnp.float64))
    np.testing.assert_allclose(theta[0], case["theta"], rtol=0., atol=1e-12)
    np.testing.assert_allclose(flux[0], case["fluxout"], rtol=0., atol=1e-10)
    assert theta.shape == flux.shape == (3, 2)
    assert bottom.shape == (3,)
    np.testing.assert_allclose(bottom, 0., rtol=0., atol=1e-10)


def test_drainage_fixtures_match_live_native_objects(tmp_path):
    reference = ROOT.parent / "aquacrop-fortran/src"
    objects = sorted(p for p in reference.glob("*.o") if p.name != "aquacrop.o")
    if not shutil.which("gcc") or not objects or not find_library("gfortran"):
        pytest.skip("Optional native probe requires gcc and prebuilt reference objects")
    probe = tmp_path / "probe"
    subprocess.run(["gcc", "-O2", "-no-pie", "-o", str(probe),
        str(ROOT / "scripts/drainage_probe.c"), *map(str, objects), "-l:libgfortran.so.5", "-lm"],
        check=True, capture_output=True, text=True)
    for case in CASES:
        text = f"{len(case['thickness'])} 46 29 {case['tau']:.17g} {case['ksat']:.17g} {case['gravel_volume_pct']:.17g}\n"
        text += "\n".join(f"{d:.17g} {t:.17g} {a * 100.:.17g}"
            for d, t, a in zip(case["thickness"], case["initial_theta"], case["adjusted_fc"])) + "\n"
        result = subprocess.run([str(probe)], input=text, text=True, capture_output=True, check=True)
        data = np.asarray([list(map(float, line.split())) for line in result.stdout.splitlines()])
        expected = np.asarray(list(zip(case["theta"], case["fluxout"])) +
                              [(case["drainage"], case["residual"])])
        np.testing.assert_allclose(data, expected, rtol=0., atol=1e-10)

    # Exercise geometries through Ottawa's 12 compartments using fresh native
    # results, rather than fixtures derived from the JAX implementation.
    generator = np.random.default_rng(41073)
    for count in (1, 2, 3, 4, 12):
        for repetition in range(4):
            dz = generator.uniform(.05, .4, count)
            initial = generator.uniform(.13, .46, count)
            adjusted = np.sort(generator.uniform(.29, .44, count))
            tau = (.1, .5, .8, 1.)[repetition]
            ksat = (.2, 3., 100., 1200.)[repetition]
            gravel = (0., 20., 0., 20.)[repetition]
            text = f"{count} 46 29 {tau:.17g} {ksat:.17g} {gravel:.17g}\n"
            text += "\n".join(f"{d:.17g} {t:.17g} {a * 100.:.17g}"
                              for d, t, a in zip(dz, initial, adjusted)) + "\n"
            result = subprocess.run([str(probe)], input=text, text=True, capture_output=True, check=True)
            native = np.asarray([list(map(float, line.split())) for line in result.stdout.splitlines()])
            theta, flux, bottom = vertical_drainage(initial, dz, [gravel] * count, adjusted,
                                                    46., 29., tau, ksat)
            np.testing.assert_allclose(theta, native[:-1, 0], rtol=0., atol=1e-12)
            np.testing.assert_allclose(flux, native[:-1, 1], rtol=0., atol=1e-10)
            np.testing.assert_allclose(bottom, native[-1, 0], rtol=0., atol=1e-10)
            residual = np.sum((initial - np.asarray(theta)) * 1000. * dz * (1. - gravel / 100.)) - float(bottom)
            assert abs(residual) < 1e-10
