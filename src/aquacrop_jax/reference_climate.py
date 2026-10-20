"""Native monthly climatology and file-based 365-day reference temperatures."""
import calendar
from datetime import date, timedelta
from typing import NamedTuple

import jax.numpy as jnp
from jax import lax
import numpy as np

from .rounding import round_fortran


class MonthlyReferenceTemperature(NamedTuple):
    temperature: np.ndarray
    reference_year: int


def reference_monthly_temperature(series):
    """Daily-record branch of CreateTnxReferenceFile, including partial months.

    Means are accumulated per observed month/year before averaging years.
    Like the source, an unfinished final month is not flushed at EOF, while
    its year still counts in the monthly divisor. Output uses file precision.
    Parsing and calendar grouping are host operations outside the JIT loop.
    """
    if series.data_type != 1:
        raise NotImplementedError("Reference climatology currently requires daily records")
    values = np.asarray(series.values, np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values):
        raise ValueError("Reference temperature requires nonempty Tmin/Tmax records")
    first = date(series.first_year, series.first_month, series.first_day)
    last = first + timedelta(days=len(values) - 1)
    years = np.full(12, last.year - first.year + 1, dtype=int)
    if last.year > first.year:
        years[:first.month - 1] -= 1
        years[last.month:] -= 1
    means = np.zeros((12, 2), np.float64)
    total = np.zeros(2, np.float64)
    days = 0
    current = first
    for temperatures in values:
        total += temperatures
        days += 1
        if current.day == calendar.monthrange(current.year, current.month)[1]:
            means[current.month - 1] += (total / days) / years[current.month - 1]
            total = np.zeros(2, np.float64)
            days = 0
        current += timedelta(days=1)
    year = round((first.year + last.year) / 2.)
    if year == 1901:
        year = 2000
    return MonthlyReferenceTemperature(
        np.asarray(round_fortran(100. * means)) / 100., year)


def daily_reference_temperature(monthly_temperature):
    """GetMonthlyTemperatureDataSetFromTnxReferenceFile + CreateTnxReference365Days.

    The monthly storage is real(sp) in the native code: its float32 round-trip
    is intentional input semantics. Interpolation is float64. Results are the
    two-decimal temperatures read back from the CLI reference file, rather
    than the separate external-temperature array interface.
    """
    monthly = jnp.asarray(monthly_temperature, jnp.float32).astype(jnp.float64)
    if monthly.shape != (12, 2):
        raise ValueError("Monthly reference temperature requires shape (12, 2)")
    month_index = jnp.asarray([month for month, length in enumerate(
        (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)) for _ in range(length)])
    within_month = jnp.asarray([day for length in
        (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
        for day in range(1, length + 1)], jnp.float64)[:, None]
    c1 = monthly[(month_index - 1) % 12] * 30.
    c2 = monthly[month_index] * 30.
    c3 = monthly[(month_index + 1) % 12] * 30.
    a = (c1 - 2. * c2 + c3) / (6. * 30. * 30. * 30.)
    b = (-6. * c1 + 9. * c2 - 3. * c3) / (6. * 30. * 30.)
    c = (11. * c1 - 7. * c2 + 2. * c3) / (6. * 30.)
    t1, t2 = 29. + within_month, 30. + within_month
    interpolated = a * (t2 * t2 * t2 - t1 * t1 * t1) + b * (t2 * t2 - t1 * t1) + c * (t2 - t1)
    # Preserve file read-back rounding: XLA otherwise replaces /100 by a
    # reciprocal multiplication, changing 98 Ottawa temperatures by one ULP.
    return round_fortran(100. * interpolated) / lax.optimization_barrier(
        jnp.asarray(100., jnp.float64))


def crop_reference_temperature(daily_temperature, first_day_of_year, *, days=365):
    """Rotate/cycle a fixed 1901 reference year to the crop's first day."""
    daily = jnp.asarray(daily_temperature, jnp.float64)
    if daily.shape != (365, 2):
        raise ValueError("Daily reference temperature requires shape (365, 2)")
    indices = (jnp.asarray(first_day_of_year, jnp.int32) - 1 + jnp.arange(days)) % 365
    return daily[indices]
