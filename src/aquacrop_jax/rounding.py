"""Integer rounding semantics of ac_utils::roundc, returned as float64."""
import jax.numpy as jnp


def round_fortran(value, *, bits=32):
    """Match roundc's clipping and absolute-epsilon half-integer neighbourhood.

    ``bits`` is a static choice (8 or 32). Returning float64 preserves the
    existing arithmetic/carry contract; callers cast to integers as needed.
    """
    if bits not in (8, 32):
        raise ValueError("roundc supports only int8 and int32")
    value = jnp.clip(jnp.asarray(value, jnp.float64),
                     -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    lower = jnp.floor(value)
    fraction = value - lower
    near_half = jnp.abs(fraction - .5) < jnp.finfo(jnp.float64).eps
    even = jnp.where(jnp.remainder(lower, 2.) == 0., lower, lower + 1.)
    # Compare the fraction directly: adding .5 could erase the very small
    # differences that roundc's epsilon branch distinguishes.
    upper = jnp.logical_or(fraction > .5,
                           jnp.logical_and(fraction == .5, value >= 0.))
    nearest = jnp.where(upper, lower + 1., lower)
    return jnp.where(near_half, even, nearest)
