import jax
import numpy as np
import pytest

from aquacrop_jax.capillary import capillary_rise, maximum_capillary_rise
from aquacrop_jax.groundwater import (adjusted_field_capacity,
    constrain_root_depth, interpolate_groundwater,
    saturate_below_water_table)
from aquacrop_jax.soil import profile_water_mm


def test_groundwater_interpolation_and_endpoint_guards():
    middle = interpolate_groundwater(15, [10, 20], [1.0, 2.0], [2.0, 4.0])
    assert float(middle.depth_m) == pytest.approx(1.5)
    assert float(middle.ec_dsm) == pytest.approx(3.0)
    assert float(interpolate_groundwater(1, [10, 20], [1.0, 2.0],
                                        [2.0, 4.0]).depth_m) == 1.0


def test_adjusted_fc_reaches_saturation_at_water_table():
    adjusted = adjusted_field_capacity(
        [0.5, 0.5, 0.5], [29.0] * 3, [46.0] * 3, 1.0)
    assert adjusted[0] > 29.0
    assert adjusted[1] > adjusted[0]
    assert adjusted[2] == pytest.approx(46.0)
    absent = adjusted_field_capacity([0.5], [29.0], [46.0], -1.0)
    assert absent == pytest.approx([29.0])


def test_water_table_saturation_and_root_constraint():
    theta = saturate_below_water_table(
        [0.2, 0.2, 0.2], [0.5, 0.5, 0.5], [46.0] * 3, 0.9)
    assert np.asarray(theta) == pytest.approx([0.2, 0.2, 0.46])
    assert float(constrain_root_depth(1.5, 0.9, 0.3)) == pytest.approx(0.9)
    assert float(constrain_root_depth(1.5, -1.0, 0.3)) == pytest.approx(1.5)


def test_maximum_capillary_rise_fortran_guards():
    assert float(maximum_capillary_rise(-0.3, 1.0, 100.0, 1.0, 1.0)) == 99.0
    assert float(maximum_capillary_rise(-0.3, 1.0, 0.0, 0.5, 1.0)) == 0.0
    assert float(maximum_capillary_rise(-0.3, 1.0, 100.0, 0.0, 5.0)) == 0.0


def test_capillary_rise_conserves_profile_water_and_jits():
    operation = jax.jit(capillary_rise)
    initial = np.asarray([0.20, 0.20, 0.20])
    result = operation(initial, [0.2] * 3, [0.0] * 3, [29.0] * 3,
                       [29.0] * 3, [13.0] * 3, [100.0] * 3,
                       [-0.3] * 3, [1.0] * 3, 1.0)
    before = profile_water_mm(initial, [0.2] * 3, [0.0] * 3)
    after = profile_water_mm(result.theta, [0.2] * 3, [0.0] * 3)
    assert float(after - before) == pytest.approx(float(result.water_mm))
    assert float(result.water_mm) >= 0.0
    assert np.asarray(result.theta) == pytest.approx(
        initial + np.asarray(result.compartment_mm) / 200.0)
