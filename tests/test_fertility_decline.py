"""Native fertility canopy decline, including its mixed-clock guards."""
from ctypes.util import find_library
import json
from pathlib import Path
import shutil
import subprocess
import sys

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pytest

from aquacrop_jax.canopy import fertility_canopy_no_water_stress

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/fertility_decline_fortran73.json").read_text())
CASES = FIXTURE["cases"]


def evaluate(row):
    return fertility_canopy_no_water_stress(row[1], row[16],
        tuple(row[2:6]), tuple(row[6:10]), *row[10:16], *row[17:21],
        gdd_mode=row[0] == 0)


def test_fertility_decline_fixture_matches_live_native_routine(tmp_path):
    if (not shutil.which("gfortran") or not find_library("gfortran") or
            not (ROOT.parent / "aquacrop-fortran/src/global.o").exists()):
        pytest.skip("Native decline probe requires gfortran and reference objects")
    output = tmp_path / "decline.json"
    result = subprocess.run([sys.executable,
        str(ROOT / "scripts/generate_fertility_decline_oracle.py"), "--output", str(output)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    actual = json.loads(output.read_text())
    for observed, expected in zip(actual["cases"], CASES, strict=True):
        assert observed["input"] == expected["input"]
        assert observed["cover"] == expected["cover"]


def test_fertility_decline_matches_native_eager_jit_and_scan():
    inputs = jnp.asarray([case["input"] for case in CASES], jnp.float64)
    expected = np.asarray([case["cover"] for case in CASES])

    def step(carry, row):
        return carry, evaluate(row)

    for actual in (jax.vmap(evaluate)(inputs), jax.jit(jax.vmap(evaluate))(inputs),
                   jax.jit(lambda x: lax.scan(step, (), x)[1])(inputs)):
        np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-14)
        np.testing.assert_array_equal(np.asarray(actual)[expected == 0.], 0.)
        assert actual.dtype == jnp.float64
        assert np.all(np.isfinite(actual))
