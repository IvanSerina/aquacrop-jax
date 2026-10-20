"""Reference climate from real Ottawa inputs, including native file precision."""
from datetime import date, timedelta
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aquacrop_jax.legacy import ClimateSeries, read_climate_series
from aquacrop_jax.reference_climate import (reference_monthly_temperature,
    daily_reference_temperature, crop_reference_temperature)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/fertility_setup_fortran73.json").read_text())


def test_ottawa_reference_climate_matches_native_monthly_and_daily_files():
    series = read_climate_series(ROOT.parent / "aquacrop-fortran/testcase/DATA/Ottawa.Tnx", 2)
    reference = reference_monthly_temperature(series)
    assert reference.reference_year == FIXTURE["reference_year"]
    np.testing.assert_array_equal(reference.temperature, FIXTURE["reference_monthly_temperature"])
    for evaluate in (daily_reference_temperature, jax.jit(daily_reference_temperature)):
        daily = evaluate(reference.temperature)
        np.testing.assert_array_equal(daily, FIXTURE["reference_daily_temperature"])
        assert daily.dtype == jnp.float64
    # Both stages are traced together, so file-precision equality must survive
    # fusion as well as their independent eager/JIT calls.
    crop = jax.jit(lambda monthly: crop_reference_temperature(
        daily_reference_temperature(monthly), 141))(reference.temperature)
    np.testing.assert_array_equal(crop, FIXTURE["crop_reference_temperature"])


def test_monthly_mean_gives_each_year_equal_weight_even_with_leap_february():
    first, last = date(2019, 1, 1), date(2020, 12, 31)
    values = []
    current = first
    while current <= last:
        low = 0. if current.year == 2019 else 10.
        values.append((low, low + 10.))
        current += timedelta(days=1)
    actual = reference_monthly_temperature(ClimateSeries(1, 1, 1, 2019, tuple(values)))
    assert actual.reference_year == 2020  # Native half-even midpoint year.
    np.testing.assert_array_equal(actual.temperature, np.tile([5., 15.], (12, 1)))


def test_partial_initial_month_counts_but_unfinished_final_month_is_not_flushed():
    series = ClimateSeries(1, 16, 1, 2020, ((2., 12.),) * 26)
    actual = reference_monthly_temperature(series)
    expected = np.zeros((12, 2))
    expected[0] = [2., 12.]
    np.testing.assert_array_equal(actual.temperature, expected)
    annual = reference_monthly_temperature(ClimateSeries(1, 1, 1, 1901, ((2., 12.),) * 365))
    assert annual.reference_year == 2000


def test_reference_interpolation_constant_months_and_annual_cycle():
    monthly = jnp.tile(jnp.asarray([7., 17.]), (12, 1))
    daily = jax.jit(daily_reference_temperature)(monthly)
    np.testing.assert_array_equal(daily, np.tile([7., 17.], (365, 1)))
    numbered = jnp.arange(730, dtype=jnp.float64).reshape((365, 2))
    crop = jax.jit(lambda values: crop_reference_temperature(values, 365, days=367))(numbered)
    np.testing.assert_array_equal(crop, np.asarray(numbered)[(364 + np.arange(367)) % 365])


def test_reference_climate_rejects_unsupported_records_and_wrong_static_shapes():
    with pytest.raises(NotImplementedError, match="daily records"):
        reference_monthly_temperature(ClimateSeries(2, 1, 1, 2020, ((1., 2.),)))
    with pytest.raises(ValueError, match="nonempty"):
        reference_monthly_temperature(ClimateSeries(1, 1, 1, 2020, ()))
    with pytest.raises(ValueError, match="12, 2"):
        daily_reference_temperature(jnp.zeros((11, 2)))
    with pytest.raises(ValueError, match="365, 2"):
        crop_reference_temperature(jnp.zeros((364, 2)), 1)
