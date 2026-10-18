"""Native fertility response parity, excluding its daily persistent state."""
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

from aquacrop_jax.fertility import soil_fertility_effect

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "tests/fixtures/fertility_fortran73.json").read_text())["cases"]
INPUTS = ("stress_level_pct", "shape_growth", "shape_cover",
          "shape_water_productivity", "shape_canopy_decline")
OUTPUTS = ("reduction_growth_pct", "reduction_cover_pct",
           "reduction_water_productivity_pct", "canopy_decline",
           "reduction_stomatal_pct")


def test_soil_fertility_effect_native_eager_jit_scan():
    args = [jnp.asarray([case[key] for case in CASES], jnp.float64) for key in INPUTS]
    expected = np.asarray([[case[key] for key in OUTPUTS] for case in CASES])
    def operation(*x):
        return soil_fertility_effect(*x)
    def body(carry, forcing):
        return carry, operation(*forcing)
    actuals = (operation(*args), jax.jit(operation)(*args),
               jax.jit(lambda xs: lax.scan(body, (), xs)[1])(tuple(args)))
    for actual in actuals:
        matrix = np.column_stack(actual)
        np.testing.assert_allclose(matrix, expected, rtol=0., atol=1e-14)
        for index in (0, 1, 2, 4):
            np.testing.assert_array_equal(matrix[:, index], expected[:, index])
        assert np.all(np.isfinite(matrix))


def test_fertility_fixture_matches_live_native_objects(tmp_path):
    reference = ROOT.parent / "aquacrop-fortran/src"
    objects = sorted(p for p in reference.glob("*.o") if p.name != "aquacrop.o")
    if not shutil.which("gcc") or not objects or not find_library("gfortran"):
        pytest.skip("Optional native probe requires gcc and prebuilt reference objects")
    probe = tmp_path / "fertility-probe"
    subprocess.run(["gcc", "-O2", "-no-pie", "-o", str(probe),
        str(ROOT / "scripts/fertility_probe.c"), *map(str, objects),
        "-l:libgfortran.so.5", "-lm"], check=True, capture_output=True)
    text = str(len(CASES)) + "\n" + "\n".join(
        " ".join(str(case[k]) for k in INPUTS) for case in CASES) + "\n"
    result = subprocess.run([str(probe)], input=text, text=True,
                            capture_output=True, check=True)
    actual = np.asarray([list(map(float, line.split())) for line in result.stdout.splitlines()])
    expected = np.asarray([[case[key] for key in OUTPUTS] for case in CASES])
    np.testing.assert_array_equal(actual, expected)
