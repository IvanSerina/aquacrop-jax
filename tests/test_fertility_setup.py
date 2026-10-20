"""Native Ottawa fertility setup, inverse polynomial and loaded calendar."""
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

from aquacrop_jax.climate import _slice, calendar_days_for_gdd
from aquacrop_jax.fertility import (days_to_reach_cover, fertility_biomass_fraction,
                                   fit_fertility_biomass_response,
                                   soil_fertility_effect, time_to_max_canopy_sf)
from aquacrop_jax.kernels import degrees_day
from aquacrop_jax.legacy import read_annotated_values, read_climate_series, read_project
from aquacrop_jax.legacy_inputs import field_fertility_stress
from aquacrop_jax.rounding import round_fortran

ROOT = Path(__file__).resolve().parents[1]
TESTCASE = ROOT.parent / "aquacrop-fortran/testcase"
FIXTURE = json.loads((ROOT / "tests/fixtures/fertility_setup_fortran73.json").read_text())
CASES = FIXTURE["cases"]
SETUP = dict(zip(FIXTURE["setup_fields"], FIXTURE["setup"], strict=True))


def test_setup_fixture_matches_live_native_initialization(tmp_path):
    reference = ROOT.parent / "aquacrop-fortran/src"
    if (not shutil.which("gfortran") or not find_library("gfortran") or
            not (reference / "run.o").exists()):
        pytest.skip("Optional native setup requires gfortran and reference objects")
    output = tmp_path / "setup.json"
    result = subprocess.run([sys.executable,
        str(ROOT / "scripts/generate_fertility_setup_oracle.py"),
        "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    actual = json.loads(output.read_text())
    np.testing.assert_allclose(actual["setup"], FIXTURE["setup"], rtol=0., atol=1e-14)
    np.testing.assert_allclose(actual["timing_inputs"], FIXTURE["timing_inputs"],
                               rtol=0., atol=1e-14)
    np.testing.assert_allclose(actual["biomass_percentages"], FIXTURE["biomass_percentages"],
                               rtol=0., atol=1e-14)
    np.testing.assert_allclose(actual["fitted_coefficients"], FIXTURE["fitted_coefficients"],
                               rtol=0., atol=1e-14)
    assert actual["reference_year"] == FIXTURE["reference_year"]
    assert actual["reference_co2"] == FIXTURE["reference_co2"]
    for key in ("reference_monthly_temperature", "reference_daily_temperature",
                "crop_reference_temperature"):
        np.testing.assert_array_equal(actual[key], FIXTURE[key])
    for observed, expected in zip(actual["cases"], CASES, strict=True):
        assert (observed["stress"], observed["calibrated"]) == (expected["stress"], expected["calibrated"])
        np.testing.assert_allclose(observed["coefficients"], expected["coefficients"],
                                   rtol=0., atol=1e-14)
        assert abs(observed["fraction"] - expected["fraction"]) <= 1e-14


def test_biomass_fraction_matches_native_eager_jit_and_scan():
    forcing = tuple(jnp.asarray(column) for column in (
        [case["stress"] for case in CASES],
        *zip(*(case["coefficients"] for case in CASES)),
        [case["calibrated"] for case in CASES]))

    def step(row):
        stress, b0, b1, b2, calibrated = row
        return fertility_biomass_fraction(stress, b0, b1, b2, calibrated=calibrated)

    def scan_step(carry, row):
        return carry, step(row)

    expected = np.asarray([case["fraction"] for case in CASES])
    for actual in (jax.vmap(step)(forcing), jax.jit(jax.vmap(step))(forcing),
                   jax.jit(lambda x: lax.scan(scan_step, (), x)[1])(forcing)):
        np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-14)
        assert actual.dtype == jnp.float64
        assert np.all(np.isfinite(actual))
    assert expected[next(i for i, case in enumerate(CASES)
                         if case["stress"] == 100 and case["calibrated"])] < 0.


def test_fit_native_samples_preserves_management_fraction_parity():
    samples = jnp.asarray(FIXTURE["biomass_percentages"], jnp.float64)
    expected = np.asarray(FIXTURE["fitted_coefficients"], np.float64)
    stresses = jnp.asarray([case["stress"] for case in CASES])
    calibrated = jnp.asarray([case["calibrated"] for case in CASES])
    fractions = np.asarray([case["fraction"] for case in CASES])

    def evaluate(samples):
        coefficients = fit_fertility_biomass_response(samples)
        fraction = jax.vmap(lambda stress, enabled: fertility_biomass_fraction(
            stress, *coefficients, calibrated=enabled))(stresses, calibrated)
        return coefficients, fraction

    for coefficients, fraction in (evaluate(samples), jax.jit(evaluate)(samples)):
        # Measured CPU float64 contraction: eager coefficient errors 1/1/0
        # ULP and fused JIT 5/24/47 ULP. The actual inverse for every integer
        # management level 0..100 still satisfies the unchanged 1e-14 bound.
        np.testing.assert_array_max_ulp(np.asarray(coefficients), expected, maxulp=64)
        np.testing.assert_allclose(fraction, fractions, rtol=0., atol=1e-14)
        assert fraction.dtype == jnp.float64


def test_fit_singular_response_and_static_sample_shape():
    for evaluate in (fit_fertility_biomass_response, jax.jit(fit_fertility_biomass_response)):
        np.testing.assert_array_equal(evaluate(jnp.full(8, 100.)), [-9., -9., -9.])
        with pytest.raises(ValueError, match="eight biomass percentages"):
            evaluate(jnp.ones(7))


@pytest.mark.parametrize("coefficients,expected", [
    ((100., -1., 0.), .5),             # Exact linear bracket.
    ((120., 0., 0.), .995),            # Native StrLow is capped at 100.
    ((.0005, 0., 0.), .01),           # Flat final bracket uses BioTop=1.
])
def test_inverse_interpolation_and_degenerate_brackets(coefficients, expected):
    for evaluate in (fertility_biomass_fraction, jax.jit(fertility_biomass_fraction)):
        actual = evaluate(50, *coefficients)
        np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-14)


def test_ottawa_inputs_reproduce_native_fertility_and_calendar_setup():
    project = read_project(TESTCASE / "LIST/Ottawa.PRM")
    run = project.runs[0]
    crop = read_annotated_values(TESTCASE / "DATA/AlfOttawaGDD.CRO")

    def value(prefix):
        return next(v for key, v in crop.items() if key.startswith(prefix))

    stress = field_fertility_stress(TESTCASE / "DATA/Ottawa.MAN")
    assert stress == SETUP["management_stress_pct"]
    effect = soil_fertility_effect(stress,
        value("Shape factor for the response of canopy expansion"),
        value("Shape factor for the response of maximum canopy cover"),
        value("Shape factor for the response of crop Water Productivity"),
        value("Shape factor for the response of decline of canopy cover"))
    np.testing.assert_array_equal(effect[:3],
        [SETUP[k] for k in ("reduction_growth_pct", "reduction_cover_pct", "reduction_wp_pct")])
    np.testing.assert_allclose(effect.canopy_decline, SETUP["canopy_decline"],
                               rtol=0., atol=1e-14)
    temperature = np.asarray(_slice(read_climate_series(TESTCASE / "DATA/Ottawa.Tnx", 2),
        run.crop_first_day, run.crop_last_day, "Temperature"))
    gdd = degrees_day(value("Base temperature"), value("Upper temperature"),
                      temperature[:, 0], temperature[:, 1], 3)
    initial = (value("Soil surface covered by an individual seedling") *
               value("Number of plants per hectare") / 1e8)
    maximum = value("Maximum canopy cover")
    growth = value("CGC for GGDays")
    emergence = value("GDDays: from sowing to emergence")
    full_gdd = days_to_reach_cover(.98 * maximum, initial, maximum, growth, emergence)
    full_day = calendar_days_for_gdd(gdd, full_gdd)
    calendar_growth = (full_gdd.astype(jnp.float64) / full_day) * growth
    emergence_day = calendar_days_for_gdd(gdd, emergence)
    season_length = run.crop_last_day - run.crop_first_day + 1
    timing_inputs = (initial, calendar_growth, maximum, emergence_day,
                     full_day, season_length, 0, 0, False)
    np.testing.assert_allclose(timing_inputs, FIXTURE["timing_inputs"], rtol=0., atol=1e-14)
    timing = time_to_max_canopy_sf(*timing_inputs,
        effect.reduction_growth_pct, effect.reduction_cover_pct, stress)
    assert int(timing.full_canopy_day) == SETUP["full_canopy_sf_day"]
    assert int(round_fortran(gdd[:int(timing.full_canopy_day)].sum())) == SETUP["full_canopy_sf_gdd"]
    fraction = fertility_biomass_fraction(stress,
        *(SETUP[f"coefficient{i}"] for i in range(3)))
    np.testing.assert_allclose(fraction, SETUP["potential_biomass_fraction"],
                               rtol=0., atol=1e-14)
