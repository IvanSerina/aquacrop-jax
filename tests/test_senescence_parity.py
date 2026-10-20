"""Unrounded native onset, persistence and recovery of early water senescence."""
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

from aquacrop_jax.canopy import (EarlyVegetativeCanopyState, WaterSenescenceState,
    early_vegetative_gdd_step, early_water_senescence_gdd_step,
    leaf_thresholds_for_eto, potential_canopy_gdd, senescence_threshold_for_eto)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/senescence_fortran73.json").read_text())
CASES = FIXTURE["cases"]


def initial_state(case):
    c, s = case["canopy_state"], case["senescence_state"]
    return (EarlyVegetativeCanopyState(*map(jnp.float64, c[:3]), jnp.bool_(c[3])),
            WaterSenescenceState(*map(jnp.float64, s[:4]),
                                  jnp.bool_(s[4]), jnp.bool_(s[5])))


def advance(carry, forcing, p):
    canopy, state = carry
    total, daily, eto, actual, fc, wp, top_actual, top_fc, top_wp, topsoil = forcing
    actual, fc, wp = (jnp.where(topsoil != 0., a, b) for a, b in
                     ((top_actual, actual), (top_fc, fc), (top_wp, wp)))
    upper, lower = leaf_thresholds_for_eto(
        eto, .15, .55, p["fao_correction"], p["adjustment_factor"])
    growing, leaf, valid = early_vegetative_gdd_step(
        canopy, total, daily, 5., 600., 1920., 1920., 2200.,
        p["initial_cover"], p["maximum_cover"], p["growth"], p["decline"],
        p["reduction_growth_pct"], p["reduction_cover_pct"],
        actual, fc, wp, upper, lower, 3.)
    threshold = senescence_threshold_for_eto(eto, p["p_senescence"], state.time_gdd,
        p["fao_correction"], p["adjustment_factor"], p["beta_pct"])
    next_canopy, next_state, stress, valid = early_water_senescence_gdd_step(
        state, growing, canopy.cover, total, daily, 5., 1920., 1920.,
        p["initial_cover"], p["initial_cover"], p["maximum_cover"],
        p["maximum_cover"] * (1. - p["reduction_cover_pct"] / 100.),
        p["decline"], actual, fc, wp, eto, p["p_senescence"],
        p["senescence_shape"], p["delay_eto_mm"], canopy_valid=valid,
        fao_correction=p["fao_correction"], adjustment_factor=p["adjustment_factor"],
        beta_pct=p["beta_pct"])
    row = jnp.stack((*next_canopy, *next_state, leaf, stress, threshold))
    return (next_canopy, next_state), (row, valid)


def trajectory(carry, forcing, parameters):
    return lax.scan(lambda state, row: advance(state, row, parameters), carry, forcing)


compiled_trajectory = jax.jit(trajectory)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_early_senescence_matches_native_eager_and_jit_scan(case):
    if case.get("phase") == "midseason":
        pytest.skip("mid-season fixture is exercised through its persistent canopy state")
    initial = initial_state(case)
    p = {key: jnp.asarray(value, jnp.bool_ if key == "fao_correction" else jnp.float64)
         for key, value in case["parameters"].items()}
    forcing = jnp.asarray(case["forcing"], jnp.float64)
    carry, eager, flags = initial, [], []
    for row in forcing:
        carry, (values, valid) = advance(carry, row, p)
        eager.append(values)
        flags.append(valid)
    _, (compiled, compiled_valid) = compiled_trajectory(initial, forcing, p)
    expected = np.asarray(case["expected"], dtype=float)
    for actual, valid in ((jnp.stack(eager), jnp.stack(flags)), (compiled, compiled_valid)):
        assert actual.dtype == jnp.float64
        assert np.all(np.isfinite(actual))
        if not case["native_finite"]:
            # Known native zero-CDC singularity, not permission to move pSen.
            assert np.isnan(expected[0, 0])
            np.testing.assert_array_equal(valid, [False])
            continue
        np.testing.assert_array_equal(valid, True)
        # Native/JIT maxima on these 31 finite days: cover/state 2.78e-17,
        # leaf stress 1.28e-13 percentage points; senescence stress and flags
        # agree exactly. Keep the existing canopy/leaf parity tolerances.
        np.testing.assert_allclose(actual[:, [0, 1, 2, 5, 6]],
                                   expected[:, [0, 1, 2, 5, 6]], rtol=0., atol=1e-14)
        np.testing.assert_array_equal(actual[:, [3, 4, 7, 8, 9]],
                                      expected[:, [3, 4, 7, 8, 9]])
        np.testing.assert_allclose(actual[:, [10, 11]], expected[:, [10, 11]],
                                   rtol=0., atol=1e-12)
        np.testing.assert_allclose(actual[:, 12], expected[:, 12], rtol=0., atol=1e-15)


def test_midseason_water_senescence_matches_native_eager_and_jit_scan():
    case = next(case for case in CASES if case["name"] == "midseason-water-senescence")
    canopy, state = initial_state(case)
    p = {key: jnp.asarray(value, jnp.bool_ if key == "fao_correction" else jnp.float64)
         for key, value in case["parameters"].items()}
    forcing = jnp.asarray(case["forcing"], jnp.float64)

    def step(carry, row):
        previous_canopy, senescence = carry
        total, daily, eto, actual, fc, wp, top_actual, top_fc, top_wp, topsoil = row
        actual = jnp.where(topsoil != 0., top_actual, actual)
        fc = jnp.where(topsoil != 0., top_fc, fc)
        wp = jnp.where(topsoil != 0., top_wp, wp)
        baseline_cover = potential_canopy_gdd(total, 5., 1920., 2200.,
                                               .05, .95, .012, .006)
        todays_canopy = previous_canopy._replace(
            cover=baseline_cover, maximum_adjusted=baseline_cover)
        next_canopy, next_senescence, stress, valid = early_water_senescence_gdd_step(
            senescence, todays_canopy, previous_canopy.cover,
            total, daily, 5., 1920., 1920., .05, .05, .95, baseline_cover,
            p["decline"], actual, fc, wp, eto, p["p_senescence"],
            p["senescence_shape"], p["delay_eto_mm"], canopy_valid=True,
            fao_correction=bool(case["parameters"]["fao_correction"]),
            adjustment_factor=p["adjustment_factor"], beta_pct=p["beta_pct"])
        row_out = jnp.stack((*next_canopy, *next_senescence, -33., stress,
                             senescence_threshold_for_eto(
                                 eto, p["p_senescence"], senescence.time_gdd,
                                 bool(case["parameters"]["fao_correction"]),
                                 p["adjustment_factor"], p["beta_pct"])))
        return (next_canopy, next_senescence), (row_out, valid)

    eager_rows, eager_valid = [], []
    carry = (canopy, state)
    for row in forcing:
        carry, (values, valid) = step(carry, row)
        eager_rows.append(values)
        eager_valid.append(valid)
    _, (compiled_rows, compiled_valid) = jax.jit(
        lambda c, s, x: lax.scan(step, (c, s), x))(canopy, state, forcing)
    expected = np.asarray(case["expected"], dtype=float)
    for actual, valid in ((jnp.stack(eager_rows), jnp.stack(eager_valid)),
                          (compiled_rows, compiled_valid)):
        np.testing.assert_array_equal(valid, True)
        np.testing.assert_allclose(actual[:, :7], expected[:, :7], rtol=0., atol=1e-14)
        np.testing.assert_array_equal(actual[:, 7:10], expected[:, 7:10])
        np.testing.assert_allclose(actual[:, 10:12], expected[:, 10:12], rtol=0., atol=1e-12)
        np.testing.assert_allclose(actual[:, 12], expected[:, 12], rtol=0., atol=1e-15)


def test_senescence_fixture_matches_live_native_objects(tmp_path):
    reference = ROOT.parent / "aquacrop-fortran/src"
    objects = sorted(p for p in reference.glob("*.o") if p.name != "aquacrop.o")
    if not shutil.which("gcc") or not objects or not find_library("gfortran"):
        pytest.skip("Optional native probe requires gcc and prebuilt reference objects")
    probe = tmp_path / "senescence-probe"
    subprocess.run(["gcc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2",
        "-no-pie", "-o", str(probe), str(ROOT / "scripts/senescence_probe.c"),
        *map(str, objects), "-l:libgfortran.so.5", "-lm"], check=True, capture_output=True)
    for case in CASES:
        output = subprocess.run([str(probe)], input=case["native_input"],
                                text=True, capture_output=True, check=True)
        rows = np.asarray([list(map(float, line.split())) for line in output.stdout.splitlines()])
        expected = np.asarray(case["expected"], dtype=float)
        if case["native_finite"]:
            np.testing.assert_array_equal(rows, expected)
        else:
            # Flags downstream of native NaN/roundc are not a portable oracle.
            np.testing.assert_array_equal(np.isfinite(rows), np.isfinite(expected))


@pytest.mark.parametrize("total,upstream_valid,fc", [(5., True, 100.),
    (1920., True, 100.), (100., False, 100.), (100., True, 20.)])
def test_unsupported_phase_or_water_domain_cannot_update_state(total, upstream_valid, fc):
    canopy, state = initial_state(CASES[0])
    result = jax.jit(early_water_senescence_gdd_step)(
        state, canopy, canopy.cover, total, 10., 5., 1920., 1920.,
        .05, .05, .95, .95, .006, 20., fc, 20., 5., .65, 3., 40.,
        canopy_valid=upstream_valid)
    next_canopy, next_state, stress, valid = result
    assert not bool(valid)
    np.testing.assert_array_equal(next_canopy, canopy)
    np.testing.assert_array_equal(next_state, state)
    assert float(stress) == -9.
