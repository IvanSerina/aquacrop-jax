"""Reference-climate ET recurrence against original native daily prefixes."""
from ctypes.util import find_library
import json
from pathlib import Path
import shutil
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aquacrop_jax.kernels import degrees_day
from aquacrop_jax.reference_et import reference_crop_transpiration

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/reference_et_fortran73.json").read_text())
CASES = FIXTURE["cases"]


def test_reference_et_fixture_matches_live_native_prefixes(tmp_path):
    if (not shutil.which("gfortran") or not find_library("gfortran") or
            not (ROOT.parent / "aquacrop-fortran/src/global.o").exists()):
        pytest.skip("Native Kc probe requires gfortran and prebuilt reference objects")
    output = tmp_path / "reference-et.json"
    result = subprocess.run([sys.executable,
        str(ROOT / "scripts/generate_reference_et_oracle.py"), "--output", str(output)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    actual = json.loads(output.read_text())
    for observed, expected in zip(actual["cases"], CASES, strict=True):
        assert observed["input"] == expected["input"]
        np.testing.assert_allclose(observed["cumulative_kc"], expected["cumulative_kc"],
                                   rtol=0., atol=1e-12)


def evaluate(row):
    daily = degrees_day(row[21], row[22], row[23], row[24], 3)
    return reference_crop_transpiration(jnp.full(80, daily),
        tuple(row[4:8]), tuple(row[8:12]), *row[12:21], row[26], row[25],
        gdd_mode=row[0] == 0, initial_cover_day=row[1], initial_cover_gdd=row[2])


def test_reference_et_matches_native_seeded_regrowth_and_cold_trajectories():
    rows = jnp.asarray([case["input"] for case in CASES], jnp.float64)
    expected = np.asarray([case["cumulative_kc"] for case in CASES])
    for result in (jax.vmap(evaluate)(rows), jax.jit(jax.vmap(evaluate))(rows)):
        np.testing.assert_allclose(result.cumulative_kc, expected, rtol=0., atol=1e-12)
        np.testing.assert_allclose(result.sum_kc, expected[:, -1], rtol=0., atol=1e-12)
        # Native prefix differences recover each day's transpiration. Their
        # subtraction includes one extra float64 rounding compared with Tpot.
        native_tpot = 5. * np.diff(expected, prepend=0., axis=1)
        np.testing.assert_allclose(result.transpiration_mm, native_tpot, rtol=0., atol=1e-12)
        assert result.cumulative_kc.dtype == jnp.float64
        assert np.all(np.isfinite(result.canopy))
        assert np.all((np.asarray(result.canopy) >= 0.) & (np.asarray(result.canopy) <= 1.))
        for i, case in enumerate(CASES):
            if case["name"].endswith("zero-gdd"):
                np.testing.assert_array_equal(result.transpiration_mm[i], 0.)


def test_reference_et_empty_series_is_zero_and_shape_is_static():
    def run(gdd):
        return reference_crop_transpiration(gdd, (3, 20, 50, 65), (25, 200, 500, 650),
            .05, .95, .18, .012, .036, .006, 1.15, 11., 60., 369.41, 8.)
    empty = jax.jit(run)(jnp.zeros(0, jnp.float64))
    assert float(empty.sum_kc) == 0.
    assert empty.canopy.shape == (0,)
    with pytest.raises(ValueError, match="one-dimensional"):
        run(jnp.zeros((2, 3)))
