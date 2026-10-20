"""Native reference crop calendar in all four crop categories."""
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
from aquacrop_jax.reference_calendar import gdd_reference_calendar

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/reference_crop_calendar_fortran73.json").read_text())
CASES = FIXTURE["cases"]


def evaluate(row):
    temperature = row[3].astype(jnp.float32).astype(jnp.float64)
    daily = jnp.full(365, degrees_day(5., 30., temperature, temperature, 3))
    return gdd_reference_calendar(daily, tuple(row[4:10]), tuple(row[10:16]),
        row[21], row[23], row[19], row[17], row[16], forage_crop=row[0] == 4,
        reproductive_crop=(row[0] == 2) | (row[0] == 3), grain_crop=row[0] == 2,
        initial_cover_day=row[1], initial_cover_gdd=row[2])


def test_reference_crop_calendar_fixture_matches_live_native(tmp_path):
    if (not shutil.which("gfortran") or not find_library("gfortran") or
            not (ROOT.parent / "aquacrop-fortran/src/preparefertilitysalinity.o").exists()):
        pytest.skip("Native reference calendar requires gfortran and reference objects")
    output = tmp_path / "calendar.json"
    result = subprocess.run([sys.executable,
        str(ROOT / "scripts/generate_reference_crop_calendar_oracle.py"), "--output", str(output)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    actual = json.loads(output.read_text())
    for observed, expected in zip(actual["cases"], CASES, strict=True):
        assert observed["input"] == expected["input"]
        np.testing.assert_array_equal(observed["expected"], expected["expected"])


def test_reference_crop_calendar_matches_native_eager_and_jit():
    rows = jnp.asarray([case["input"] for case in CASES], jnp.float64)
    expected = np.asarray([case["expected"] for case in CASES])
    for actual in (jax.vmap(evaluate)(rows), jax.jit(jax.vmap(evaluate))(rows)):
        values = np.column_stack(actual[:-1])
        np.testing.assert_array_equal(values[:, :7], expected[:, :7])
        np.testing.assert_allclose(values[:, 7:], expected[:, 7:], rtol=0., atol=1e-14)
        np.testing.assert_array_equal(actual.valid, True)


def test_ottawa_reference_calendar_from_native_sp_temperature_arrays():
    fixture = json.loads((ROOT / "tests/fixtures/fertility_setup_fortran73.json").read_text())
    parameters = dict(zip(fixture["thermal_fields"], fixture["thermal_parameters"], strict=True))
    temperature = jnp.asarray(fixture["crop_reference_temperature"], jnp.float32).astype(jnp.float64)
    daily = degrees_day(5., 30., temperature[:, 0], temperature[:, 1], 3)
    thermal = tuple(parameters[k] for k in ("emergence_gdd", "full_canopy_gdd", "senescence_gdd", "maturity_gdd")) + (0, 0)
    args = (daily, (1, 39, 164, 164, 0, 0), thermal, .012, .006, .95, 100, 118)
    for operation in (gdd_reference_calendar, jax.jit(gdd_reference_calendar)):
        actual = operation(*args, forage_crop=True)
        np.testing.assert_array_equal(actual[:7], fixture["reference_calendar"][:7])
        np.testing.assert_allclose(actual[7:10], fixture["reference_calendar"][7:10], rtol=0., atol=1e-14)
        assert bool(actual.valid)


def test_reference_crop_calendar_signals_no_heat_and_static_shape_errors():
    actual = jax.jit(gdd_reference_calendar)(jnp.zeros(365), (1, 39, 164, 164, 0, 0),
        (5, 461, 1803, 1803, 0, 0), .012, .006, .95, 100, 118, forage_crop=True)
    assert not bool(actual.valid)
    with pytest.raises(ValueError, match="six calendar/thermal"):
        gdd_reference_calendar(jnp.ones(365), (1, 2), (1, 2), .01, .01, .95, 50, 10)
