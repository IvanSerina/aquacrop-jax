"""Native potential ET parity, including pre-emergence and cold guards."""
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

from aquacrop_jax.canopy import calculate_potential_et

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/et_fortran73.json").read_text())
CASES = FIXTURE["cases"]


def test_potential_et_native_eager_jit_scan():
    inputs = {key: jnp.asarray([c["inputs"][key] for c in CASES])
              for key in FIXTURE["input_order"]}
    expected = np.asarray([[c["transpiration_mm"], c["evaporation_mm"]] for c in CASES])
    def body(carry, forcing):
        return carry, calculate_potential_et(**forcing)
    results = (calculate_potential_et(**inputs),
               jax.jit(calculate_potential_et)(**inputs),
               jax.jit(lambda x: lax.scan(body, (), x)[1])(inputs))
    for result in results:
        actual = np.column_stack((result.transpiration_mm, result.evaporation_mm))
        assert actual.dtype == np.float64
        # The unchanged smooth branches differ by <= 2e-15 mm/day. This absolute
        # float64 margin is separate from the exact zero branch assertions.
        np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-12)
        np.testing.assert_array_equal(actual[expected == 0.], 0.)
        assert np.all(np.isfinite(actual))
        assert np.all(actual >= 0.)
    before = next(c for c in CASES if c["name"] == "before-half-percent-limit1")
    assert before["transpiration_mm"] == 0.
    assert before["evaporation_mm"] == 5.5
    cold = next(c for c in CASES if c["name"] == "tiny-canopy-cold-limit1")
    assert cold["transpiration_mm"] > 0.


def test_et_fixture_matches_live_native_objects(tmp_path):
    reference = ROOT.parent / "aquacrop-fortran/src"
    objects = sorted(p for p in reference.glob("*.o") if p.name != "aquacrop.o")
    if not shutil.which("gcc") or not objects or not find_library("gfortran"):
        pytest.skip("Optional native probe requires gcc and prebuilt reference objects")
    probe = tmp_path / "et-probe"
    subprocess.run(["gcc", "-O2", "-no-pie", "-o", str(probe),
        str(ROOT / "scripts/et_probe.c"), *map(str, objects),
        "-l:libgfortran.so.5", "-lm"], check=True, capture_output=True)
    text = str(len(CASES)) + "\n" + "\n".join(
        " ".join(str(c["inputs"][key]) for key in FIXTURE["input_order"])
        for c in CASES) + "\n"
    native = subprocess.run([str(probe)], input=text, text=True,
                            capture_output=True, check=True)
    actual = np.asarray([list(map(float, line.split())) for line in native.stdout.splitlines()])
    expected = np.asarray([[c["transpiration_mm"], c["evaporation_mm"]] for c in CASES])
    np.testing.assert_array_equal(actual, expected)
