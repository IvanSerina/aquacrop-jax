import jax.numpy as jnp
import pytest
from aquacrop_jax.kernels import biomass_increment, curve_number_runoff, degrees_day


@pytest.mark.parametrize("method,expected", [(1, 11.35), (2, 11.35), (3, 11.35)])
def test_degrees_day_regular(method, expected):
    assert float(degrees_day(5.0, 30.0, 10.0, 22.7, method)) == pytest.approx(expected)


def test_degrees_day_fortran_method_branches():
    assert float(degrees_day(5.0, 30.0, -10.0, 40.0, 1)) == pytest.approx(10.0)
    assert float(degrees_day(5.0, 30.0, -10.0, 40.0, 2)) == pytest.approx(12.5)
    assert float(degrees_day(5.0, 30.0, -10.0, 40.0, 3)) == pytest.approx(5.0)


def test_runoff_and_biomass_guards():
    runoff = curve_number_runoff(0.1, 46)
    biomass = biomass_increment(16.1, 4.0, 0.0)
    assert runoff.dtype == jnp.float64
    assert biomass.dtype == jnp.float64
    assert float(runoff) == 0.0
    assert float(biomass) == 0.0
