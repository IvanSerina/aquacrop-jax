"""Reference calendar semantics differ deliberately from the simulation clock."""
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

from aquacrop_jax.kernels import degrees_day
from aquacrop_jax.reference_climate import reference_calendar_days

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/reference_calendar_fortran73.json").read_text())
CASES = FIXTURE["cases"]


def evaluate(row):
    indices = jnp.arange(365)
    temperatures = jnp.where((row[2] == 1) & (indices % 2 == 1), row[4], row[3])
    temperatures = temperatures.at[0].set(jnp.where(row[2] == 2, row[4], row[3]))
    temperatures = temperatures.at[364].set(row[5]).astype(jnp.float32).astype(jnp.float64)
    gdd = degrees_day(5., 30., temperatures, temperatures, 3)
    return reference_calendar_days(gdd, row[0], start_offset=row[1])


def test_reference_calendar_fixture_matches_live_native(tmp_path):
    if (not shutil.which("gfortran") or not find_library("gfortran") or
            not (ROOT.parent / "aquacrop-fortran/src/global.o").exists()):
        pytest.skip("Native calendar probe requires gfortran and reference objects")
    output = tmp_path / "calendar.json"
    result = subprocess.run([sys.executable,
        str(ROOT / "scripts/generate_reference_calendar_oracle.py"), "--output", str(output)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["cases"] == CASES


def test_reference_calendar_matches_native_eager_jit_and_scan():
    inputs = jnp.asarray([case["input"] for case in CASES], jnp.float64)
    expected = np.asarray([case["days"] for case in CASES])

    def step(carry, row):
        return carry, evaluate(row)

    for actual in (jax.vmap(evaluate)(inputs), jax.jit(jax.vmap(evaluate))(inputs),
                   jax.jit(lambda x: lax.scan(step, (), x)[1])(inputs)):
        np.testing.assert_array_equal(actual.days, expected)
        np.testing.assert_array_equal(actual.valid, True)
        assert actual.days.dtype == jnp.int32
    midpoint = next(i for i, case in enumerate(CASES)
                    if case["name"] == "constant-20.0-target-10")
    assert expected[midpoint] == 0  # Overshoot=0.5, roundc half-even=0.
    sparse = next(i for i, case in enumerate(CASES)
                  if case["name"] == "sparse-heat-skip-last-target-10")
    assert expected[sparse] == 3277  # Ten visits to element1; element365 is skipped.


def test_no_reference_heat_is_invalid_instead_of_a_native_infinite_loop():
    for operation in (reference_calendar_days, jax.jit(reference_calendar_days)):
        result = operation(jnp.zeros(365), 5)
        assert not bool(result.valid)
        assert int(operation(jnp.zeros(365), 0).days) == 0
    with pytest.raises(ValueError, match="365 daily"):
        reference_calendar_days(jnp.ones(364), 10)
