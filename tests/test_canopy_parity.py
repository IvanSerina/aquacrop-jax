"""Potential canopy guards against unrounded native Fortran outputs."""
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

from aquacrop_jax.canopy import potential_canopy_calendar, potential_canopy_gdd

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "tests/fixtures/canopy_fortran73.json").read_text())["cases"]


def arguments(case):
    args = tuple(case[k] for k in ("emergence", "senescence", "maturity",
        "initial_cover", "maximum_cover", "growth", "decline"))
    kwargs = dict(fertility_growth_reduction_pct=case["reduction_growth_pct"],
                  fertility_cover_reduction_pct=case["reduction_cover_pct"])
    if case["mode"] == 1:
        kwargs["delayed_days"] = case["delayed_days"]
    return args, kwargs


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_potential_canopy_native_guards_eager_jit_scan(case):
    operation = potential_canopy_gdd if case["mode"] == 0 else potential_canopy_calendar
    args, kwargs = arguments(case)
    times = jnp.asarray(case["times"], jnp.float64)
    expected = np.asarray(case["cover"])
    # The unchanged smooth branches agree within 2e-16 on these native cases.
    # 1e-14 cover fraction allows float64 exp/fusion differences, with exact
    # zero assertions independently enforcing every native guard.
    def body(carry, time):
        return carry, operation(time, *args, **kwargs)
    results = (operation(times, *args, **kwargs),
               jax.jit(operation)(times, *args, **kwargs),
               jax.jit(lambda t: lax.scan(body, (), t)[1])(times))
    for result in results:
        assert result.dtype == jnp.float64
        np.testing.assert_allclose(result, expected, rtol=0., atol=1e-14)
        np.testing.assert_array_equal(np.asarray(result)[expected == 0.], 0.)
        assert np.all(np.asarray(result)[expected > 0.] > 0.)
        assert np.all(np.isfinite(result))


def test_canopy_fixtures_match_live_native_objects(tmp_path):
    reference = ROOT.parent / "aquacrop-fortran/src"
    objects = sorted(p for p in reference.glob("*.o") if p.name != "aquacrop.o")
    if not shutil.which("gcc") or not objects or not find_library("gfortran"):
        pytest.skip("Optional native probe requires gcc and prebuilt reference objects")
    probe = tmp_path / "canopy-probe"
    subprocess.run(["gcc", "-O2", "-no-pie", "-o", str(probe),
        str(ROOT / "scripts/canopy_probe.c"), *map(str, objects),
        "-l:libgfortran.so.5", "-lm"], check=True, capture_output=True)
    for case in CASES:
        fields = [case["mode"], case["delayed_days"], len(case["times"])]
        fields += [case[k] for k in ("emergence", "senescence", "maturity",
            "initial_cover", "maximum_cover", "growth", "decline",
            "reduction_growth_pct", "reduction_cover_pct")]
        text = " ".join(map(str, fields)) + "\n" + " ".join(map(str, case["times"])) + "\n"
        native = subprocess.run([str(probe)], input=text, text=True,
                                capture_output=True, check=True)
        cover = [float(line.split()[1]) for line in native.stdout.splitlines()]
        np.testing.assert_array_equal(cover, case["cover"])
