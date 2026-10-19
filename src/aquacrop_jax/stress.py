"""Shared, dependency-free water and fertility stress response."""

import jax.numpy as jnp

from .rounding import round_fortran


F64 = jnp.float64


def stress_coefficient(relative_depletion, upper_limit, lower_limit,
                       shape_factor):
    """Exact functional form of ``global.f90::KsAny``."""
    values = [jnp.asarray(v, F64) for v in
              (relative_depletion, upper_limit, lower_limit, shape_factor)]
    depletion, upper, lower, shape = values
    upper = jnp.where(lower - upper < 1e-4, lower - 1e-4, upper)
    relative = (depletion - upper) / (lower - upper)
    effective_shape = jnp.where(jnp.abs(round_fortran(10.0 * shape)) == 0.0,
                                jnp.asarray(0.01, F64), shape)
    # Preserve the source operation order (exp(x)-1); the small difference
    # from expm1 is observable in the unrounded fertility canopy-decline term.
    curved = 1.0 - (jnp.exp(relative * effective_shape) - 1.0) / (
        jnp.exp(effective_shape) - 1.0)
    return jnp.where(relative <= jnp.finfo(F64).eps, jnp.ones_like(curved),
                     jnp.where(relative >= 1.0, jnp.zeros_like(curved),
                               jnp.clip(curved, 0.0, 1.0)))
