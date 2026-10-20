"""Sequential thermal time over a native daily temperature-file record."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .rounding import round_fortran


class GrowingDegreeDaysResult(NamedTuple):
    value: jax.Array
    valid: jax.Array


def growing_degree_days_from_record(daily_gdd, period_days, first_offset=0, *,
                                    undefined_year=False):
    """Daily-file branch of tempprocessing.f90::GrowingDegreeDays.

    ``daily_gdd`` comes from the file's float64 temperatures and selected
    DegreesDay method. ``first_offset`` is zero based, after the caller's
    native SetDayNrToYundef date normalization for an undefined-year record.
    Preserve the strict record-end guard, skipped final array element on
    cycling, chronological sum and final roundc. Nonpositive periods return
    -9; uncovered starts return 0; exhausted finite records return -9.
    ``valid`` distinguishes these native sentinels from a valid zero-GDD sum.
    External temperature arrays and constant-temperature forcing are separate
    native branches, deliberately outside this operator.
    """
    values = jnp.asarray(daily_gdd, jnp.float64)
    period, offset = [jnp.asarray(v, jnp.int32) for v in (period_days, first_offset)]
    count = values.shape[0]
    if count < 2:
        return GrowingDegreeDaysResult(jnp.where(period > 0, 0, -9).astype(jnp.int32), jnp.bool_(False))
    covered = (offset >= 0) & (offset < count - 1)

    def accumulate(_):
        initial = (offset, offset + 1, period - 1, values[offset])
        def condition(state):
            _, day, remaining, _ = state
            return (remaining > 0) & ((day < count - 1) | jnp.asarray(undefined_year, jnp.bool_))
        def step(state):
            index, day, remaining, total = state
            next_index = jnp.where(index + 1 == count - 1, 0, index + 1)
            return next_index, day + 1, remaining - 1, total + values[next_index]
        _, _, remaining, total = lax.while_loop(condition, step, initial)
        valid = remaining == 0
        value = jnp.where(valid, round_fortran(total), -9.).astype(jnp.int32)
        return GrowingDegreeDaysResult(value, valid)

    def invalid(_):
        return GrowingDegreeDaysResult(jnp.where(period > 0, 0, -9).astype(jnp.int32), jnp.bool_(False))

    return lax.cond((period > 0) & covered, accumulate, invalid, None)
