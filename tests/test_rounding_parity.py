"""Exact roundc parity, including nearby floats and native clipping."""
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

from aquacrop_jax.rounding import round_fortran
from aquacrop_jax.salinity import round_fortran as salinity_round

ROOT = Path(__file__).resolve().parents[1]
RECORDS = json.loads((ROOT / "tests/fixtures/rounding_fortran73.json").read_text())["records"]


@pytest.mark.parametrize("bits", [8, 32])
def test_roundc_matches_native_exactly_eager_jit_scan(bits):
    values = jnp.asarray([r["value"] for r in RECORDS], jnp.float64)
    expected = np.asarray([r[f"int{bits}"] for r in RECORDS])
    def operation(x):
        return round_fortran(x, bits=bits)
    def body(carry, value):
        return carry, operation(value)
    for actual in (operation(values), jax.jit(operation)(values),
                   jax.jit(lambda x: lax.scan(body, (), x)[1])(values)):
        assert actual.dtype == jnp.float64
        np.testing.assert_array_equal(actual, expected)
    assert salinity_round is round_fortran


def test_rounding_fixture_matches_live_native_objects(tmp_path):
    reference = ROOT.parent / "aquacrop-fortran/src"
    objects = sorted(p for p in reference.glob("*.o") if p.name != "aquacrop.o")
    if not shutil.which("gcc") or not objects or not find_library("gfortran"):
        pytest.skip("Optional native probe requires gcc and prebuilt reference objects")
    probe = tmp_path / "rounding-probe"
    subprocess.run(["gcc", "-O2", "-no-pie", "-o", str(probe),
        str(ROOT / "scripts/rounding_probe.c"), *map(str, objects),
        "-l:libgfortran.so.5", "-lm"], check=True, capture_output=True)
    text = str(len(RECORDS)) + "\n" + "\n".join(repr(r["value"]) for r in RECORDS) + "\n"
    result = subprocess.run([str(probe)], input=text, text=True, capture_output=True, check=True)
    actual = np.asarray([list(map(int, line.split())) for line in result.stdout.splitlines()])
    np.testing.assert_array_equal(actual, [[r["int32"], r["int8"]] for r in RECORDS])
