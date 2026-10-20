"""Native monthly climatology and file-based 365-day reference temperatures."""
import calendar
from datetime import date, timedelta
from typing import NamedTuple

import jax.numpy as jnp
import jax
from jax import lax
import numpy as np

from .rounding import round_fortran


class MonthlyReferenceTemperature(NamedTuple):
    temperature: np.ndarray
    reference_year: int


class ReferenceCalendarDays(NamedTuple):
    days: jax.Array
    valid: jax.Array


def reference_calendar_days(daily_gdd, target_gdd, *, start_offset=0):
    """Array branch of SumCalendarDaysReferenceTnx, with its native 364-day cycle.

    GDD must come from the native real(sp) reference-temperature storage,
    which differs from the CLI file-based GDD used for Bnormalized. The last
    consumed day is counted using the native overshoot ratio and roundc, not
    the ordinary simulation calendar rule. A positive target with no heat
    returns valid=False instead of reproducing the native infinite loop.
    """
    daily = jnp.asarray(daily_gdd, jnp.float64)
    if daily.shape != (365,):
        raise ValueError("Reference calendar requires 365 daily GDD values")
    target = jnp.asarray(target_gdd, jnp.int32).astype(jnp.float64)
    heat = jnp.any(daily[:364] > 0.)

    def step(state):
        remaining, count, index = state
        value = daily[index]
        safe_remaining = jnp.where(remaining > .1, remaining, 1.)
        overshoot = jnp.where(value > remaining,
            round_fortran((value - remaining) / safe_remaining) >= 1., True)
        return remaining - value, count + overshoot.astype(jnp.int32), (index + 1) % 364

    initial = (target, jnp.int32(0), jnp.asarray(start_offset, jnp.int32) % 364)
    _, count, _ = lax.while_loop(lambda s: (s[0] > .1) & heat, step, initial)
    return ReferenceCalendarDays(count, (target <= 0.) | heat)


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
