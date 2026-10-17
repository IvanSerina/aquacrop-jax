"""Daily no-bund surface entry against unrounded native Fortran fixtures."""
import json
from pathlib import Path
from ctypes.util import find_library
import shutil
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aquacrop_jax.hydrology import daily_infiltration, daily_water_step_irrigated
from aquacrop_jax.soil import profile_water_mm

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "tests/fixtures/infiltration_fortran73.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_daily_infiltration_matches_native_state_flux_rejection_and_balance(case):
    count = len(case["thickness"])
    arguments = (case["initial_theta"], case["thickness"], [case["gravel_volume_pct"]] * count,
                 case["adjusted_fc"], 46., 29., case["tau"], case["ksat"],
                 case["water_mm"], case["initial_fluxout"])
    for operation in (daily_infiltration.__wrapped__, daily_infiltration):
        result = operation(*arguments)
        assert result.theta.dtype == jnp.float64
        np.testing.assert_allclose(result.theta, case["theta"], rtol=0., atol=1e-12)
        np.testing.assert_allclose(result.fluxout, case["fluxout"], rtol=0., atol=1e-10)
        np.testing.assert_allclose([result.deep_percolation, result.rejected],
            [case["deep_percolation"], case["rejected"]], rtol=0., atol=1e-10)
        before = profile_water_mm(arguments[0], arguments[1], arguments[2])
        after = profile_water_mm(result.theta, arguments[1], arguments[2])
        residual = before + case["water_mm"] - after - result.deep_percolation - result.rejected
        assert abs(float(residual)) < 1e-10
        assert float(case["water_mm"] - result.rejected) == pytest.approx(case["accepted"], abs=1e-10)


def test_daily_irrigation_respects_surface_conductivity_before_storage():
    result = daily_water_step_irrigated([.2, .2], [.1, .1], [0., 0.], [.29, .29],
                                        46., 29., 1., 1., 0., 30., 46.)
    assert float(result.infiltration) == pytest.approx(1., abs=1e-10)
    assert float(result.runoff) == pytest.approx(29., abs=1e-10)
    assert float(result.residual) == pytest.approx(0., abs=1e-10)
    np.testing.assert_allclose(result.theta, [.21, .2], rtol=0., atol=1e-12)


def test_daily_infiltration_matches_additional_live_native_profiles(tmp_path):
    reference = ROOT.parent / "aquacrop-fortran/src"
    objects = sorted(p for p in reference.glob("*.o") if p.name != "aquacrop.o")
    if not shutil.which("gcc") or not objects or not find_library("gfortran"):
        pytest.skip("Optional native probe requires gcc and prebuilt reference objects")
    probe = tmp_path / "probe"
    subprocess.run(["gcc", "-O2", "-no-pie", "-o", str(probe),
        str(ROOT / "scripts/infiltration_probe.c"), *map(str, objects), "-l:libgfortran.so.5", "-lm"],
        check=True, capture_output=True, text=True)
    generator = np.random.default_rng(41074)
    for count in (1, 2, 3, 4, 12):
        for repetition in range(4):
            dz = generator.uniform(.05, .4, count)
            initial = generator.uniform(.13, .46, count)
            adjusted = np.sort(generator.uniform(.29, .44, count))
            tau, ksat = (.0, .1, .5, 1.)[repetition], (.2, 3., 100., 1200.)[repetition]
            gravel = (0., 20., 0., 20.)[repetition]
            water = float(generator.uniform(.1, 100.))
            initial_flux = generator.uniform(0., .8 * ksat, count)
            text = f"{count} 46 29 {tau:.17g} {ksat:.17g} {gravel:.17g} {water:.17g}\n"
            text += "\n".join(f"{d:.17g} {t:.17g} {a * 100.:.17g} {f:.17g}"
                              for d, t, a, f in zip(dz, initial, adjusted, initial_flux)) + "\n"
            native = subprocess.run([str(probe)], input=text, text=True, capture_output=True, check=True)
            lines = [list(map(float, line.split())) for line in native.stdout.splitlines()]
            result = daily_infiltration(initial, dz, [gravel] * count, adjusted, 46., 29.,
                                        tau, ksat, water, initial_flux)
            np.testing.assert_allclose(result.theta, np.asarray(lines[:-1])[:, 0], rtol=0., atol=1e-12)
            np.testing.assert_allclose(result.fluxout, np.asarray(lines[:-1])[:, 1], rtol=0., atol=1e-10)
            np.testing.assert_allclose([result.deep_percolation, result.rejected], lines[-1][:2], rtol=0., atol=1e-10)
