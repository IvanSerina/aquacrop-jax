"""JAX implementation of AquaCrop."""

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

__version__ = "0.1.0"
