import numpy as np
import pytest

from aquacrop_jax.roots import actual_rooting_depth_gdd, root_zone_water


def test_ottawa_gdd_root_depth_guards_and_maximum():
    args = dict(cycle_days=180, emergence_gdd=5, maximum_rooting_gdd=1920,
                minimum_depth=0.30, maximum_depth=3.0, shape_factor=15)
    assert float(actual_rooting_depth_gdd(0, cumulative_gdd=0.0, **args)) == 0.0
    assert float(actual_rooting_depth_gdd(1, cumulative_gdd=0.0, **args)) == pytest.approx(0.30)
    assert float(actual_rooting_depth_gdd(100, cumulative_gdd=1920.0, **args)) == pytest.approx(3.0)
    assert float(actual_rooting_depth_gdd(181, cumulative_gdd=1920.0, **args)) == 0.0


def test_root_zone_uses_fractional_boundary_compartment():
    result = root_zone_water(
        [0.29, 0.20], [0.10, 0.10], [0.0, 0.0], 0.15,
        29.0, 13.0, 46.0)
    assert np.asarray(result.fractions) == pytest.approx([1.0, 0.5])
    assert float(result.actual_mm) == pytest.approx(39.0)
    assert float(result.field_capacity_mm) == pytest.approx(43.5)
    assert float(result.wilting_point_mm) == pytest.approx(19.5)
    assert float(result.relative_depletion) == pytest.approx(0.1875)


def test_root_zone_relative_depletion_is_bounded():
    dry = root_zone_water([0.0], [0.1], [0.0], 0.1, 29.0, 13.0, 46.0)
    wet = root_zone_water([0.46], [0.1], [0.0], 0.1, 29.0, 13.0, 46.0)
    assert float(dry.relative_depletion) == pytest.approx(1.0)
    assert float(wet.relative_depletion) == pytest.approx(0.0)
