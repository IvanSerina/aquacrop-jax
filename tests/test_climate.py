from datetime import date
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from aquacrop_jax.climate import (calendar_days_for_gdd, date_to_day_number,
                                  day_number_to_date)
from aquacrop_jax.kernels import degrees_day
from aquacrop_jax.legacy import read_climate_series
from aquacrop_jax.outputs import read_daily_output


TESTCASE = Path(__file__).parents[2] / 'aquacrop-fortran/testcase'


def test_aquacrop_day_number_epoch():
    assert date_to_day_number(date(2014, 5, 21)) == 41414
    assert day_number_to_date(41414) == date(2014, 5, 21)


def test_sum_calendar_days_counts_crossing_day_and_missing_record_under_jit_scan():
    daily = jnp.asarray([0., 3., 4., 0., 5.], jnp.float64)
    targets = jnp.asarray([0., 3., 4., 12., 13.], jnp.float64)

    def body(carry, target):
        return carry, calendar_days_for_gdd(daily, target)

    expected = [0, 2, 3, 5, -9]
    np.testing.assert_array_equal(
        [int(calendar_days_for_gdd(daily, target)) for target in targets], expected)
    np.testing.assert_array_equal(
        jax.jit(lambda values: lax.scan(body, (), values)[1])(targets), expected)


def test_ottawa_native_emergence_occurs_on_first_sowing_day():
    native = read_daily_output(TESTCASE / 'OUTP_REF/OttawaPRMday.OUT').records
    assert [(int(record.mapping()['dap']), int(record.mapping()['stage']))
            for record in native[:2]] == [(1, 1), (2, 2)]
    series = read_climate_series(TESTCASE / 'DATA/Ottawa.Tnx', 2)
    first = date_to_day_number(date(series.first_year, series.first_month, series.first_day))
    temperatures = np.asarray(series.values[41414 - first:41414 - first + 2])
    gdd = degrees_day(5., 30., temperatures[:, 0], temperatures[:, 1], 3)
    assert np.isclose(float(gdd[0]), 11.35, rtol=0., atol=1e-12)
    assert int(calendar_days_for_gdd(gdd, 5)) == 1
