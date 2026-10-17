import numpy as np
import pytest

from aquacrop_jax.transpiration import (compartment_transpiration,
                                       root_sink_profile,
                                       stress_coefficient)


def test_stress_coefficient_guards_and_linear_midpoint():
    assert float(stress_coefficient(0.2, 0.5, 1.0, 0.0)) == 1.0
    assert float(stress_coefficient(1.0, 0.5, 1.0, 0.0)) == 0.0
    assert float(stress_coefficient(0.75, 0.5, 1.0, 0.0)) == pytest.approx(
        0.5012499974)


def test_root_sink_profile_fractional_boundary():
    fraction, sink = root_sink_profile([0.1, 0.1, 0.1], 0.25, 0.048, 0.012)
    assert np.asarray(fraction) == pytest.approx([1.0, 1.0, 0.5])
    assert np.asarray(sink) == pytest.approx([0.0408, 0.0264, 0.0156])


def test_transpiration_is_conservative_and_excludes_unrooted_soil():
    result = compartment_transpiration(
        theta=[0.29, 0.29, 0.29], thickness=[0.1, 0.1, 0.1],
        gravel_volume_pct=[0.0, 0.0, 0.0],
        field_capacity_pct=[29.0, 29.0, 29.0],
        wilting_point_pct=[13.0, 13.0, 13.0],
        saturation_pct=[46.0, 46.0, 46.0], rooting_depth=0.2,
        potential_mm=5.0, p_stomatal=0.65, smax_top=0.048,
        smax_bottom=0.012)
    assert float(result.actual_mm) == pytest.approx(5.0)
    assert sum(np.asarray(result.extraction_mm)) == pytest.approx(5.0)
    assert np.asarray(result.theta)[2] == pytest.approx(0.29)
    before = 1000.0 * 0.1 * 3 * 0.29
    after = 1000.0 * 0.1 * sum(np.asarray(result.theta))
    assert before - after == pytest.approx(float(result.actual_mm))


def test_dry_compartments_limit_actual_transpiration():
    result = compartment_transpiration(
        theta=[0.13, 0.13], thickness=[0.1, 0.1],
        gravel_volume_pct=[0.0, 0.0], field_capacity_pct=[29.0, 29.0],
        wilting_point_pct=[13.0, 13.0], saturation_pct=[46.0, 46.0],
        rooting_depth=0.2, potential_mm=5.0, p_stomatal=0.65,
        smax_top=0.048, smax_bottom=0.012)
    assert float(result.actual_mm) == 0.0
    assert np.asarray(result.theta) == pytest.approx([0.13, 0.13])
