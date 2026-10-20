"""Bnormalized calibration recurrence against actual native daily prefixes."""
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
from aquacrop_jax.reference_biomass import normalized_biomass_trajectory
from aquacrop_jax.reference_et import reference_crop_transpiration

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/normalized_biomass_fortran73.json").read_text())
CASES = FIXTURE["cases"]


def evaluate(row):
    gdd = jnp.full(80, degrees_day(row[21], row[22], row[23], row[24], 3))
    args = (gdd, tuple(row[4:8]), tuple(row[8:12]), *row[12:21], row[26], row[25])
    calendar = dict(gdd_mode=row[0] == 0,
                    initial_cover_day=row[1], initial_cover_gdd=row[2],
                    senescence_exponent=row[44])
    baseline = reference_crop_transpiration(*args, **calendar).sum_kc
    sum_kc_top = jnp.where(row[43] >= 0., row[43], baseline)
    result = normalized_biomass_trajectory(*args, **calendar, sum_kc_top=sum_kc_top,
        fertility_stress_pct=row[27], reduction_growth_pct=row[28],
        reduction_cover_pct=row[29], reduction_wp_pct=row[30], canopy_decline=row[31],
        full_canopy_sf_day=row[32], full_canopy_sf_gdd=row[33], days_per_gdd=row[34],
        water_productivity=row[35], wp_yield_pct=row[36], yield_formation_days=row[37],
        switch_day=row[38], flowering_day=row[39], reproductive_crop=row[40] != 0,
        fertility_mode=row[41] != 0, reduction_stomatal_pct=row[42])
    return sum_kc_top, result


def test_normalized_biomass_fixture_matches_live_native_prefixes(tmp_path):
    if (not shutil.which("gfortran") or not find_library("gfortran") or
            not (ROOT.parent / "aquacrop-fortran/src/tempprocessing.o").exists()):
        pytest.skip("Native Bnormalized probe requires gfortran and reference objects")
    output = tmp_path / "biomass.json"
    result = subprocess.run([sys.executable,
        str(ROOT / "scripts/generate_normalized_biomass_oracle.py"), "--output", str(output)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    actual = json.loads(output.read_text())
    for observed, expected in zip(actual["cases"], CASES, strict=True):
        assert observed["input"] == expected["input"]
        assert observed["sum_kc_top"] == expected["sum_kc_top"]
        np.testing.assert_array_equal(observed["cumulative_biomass"], expected["cumulative_biomass"])


def test_normalized_biomass_matches_native_fertility_and_salinity_calibration():
    rows = jnp.asarray([case["input"] for case in CASES], jnp.float64)
    expected = np.asarray([case["cumulative_biomass"] for case in CASES])
    kc = np.asarray([case["sum_kc_top"] for case in CASES])
    for baseline, actual in (jax.vmap(evaluate)(rows), jax.jit(jax.vmap(evaluate))(rows)):
        np.testing.assert_allclose(baseline, kc, rtol=0., atol=1e-12)
        np.testing.assert_allclose(actual.cumulative_biomass, expected, rtol=0., atol=1e-12)
        np.testing.assert_allclose(actual.biomass, expected[:, -1], rtol=0., atol=1e-12)
        assert actual.biomass.dtype == jnp.float64
        for field in actual:
            assert np.all(np.isfinite(field))


def test_normalized_biomass_empty_series_and_shape_contract():
    def run(gdd):
        return normalized_biomass_trajectory(gdd, (3, 20, 50, 65), (25, 200, 500, 650),
            .05, .95, .18, .012, .036, .006, 1.15, 11., 60., 369.41, 8.,
            sum_kc_top=1., full_canopy_sf_day=25, full_canopy_sf_gdd=250)
    empty = jax.jit(run)(jnp.zeros(0, jnp.float64))
    assert float(empty.biomass) == 0.
    assert empty.cumulative_biomass.shape == (0,)
    with pytest.raises(ValueError, match="one-dimensional"):
        run(jnp.zeros((2, 3)))
