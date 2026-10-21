import numpy as np
import pytest
import jax
import jax.numpy as jnp

from aquacrop_jax.roots import (actual_rooting_depth_calendar,
                                actual_rooting_depth_gdd, canopy_water_zone,
                                root_zone_water)


def test_ottawa_gdd_root_depth_guards_and_maximum():
    args = dict(cycle_days=180, emergence_gdd=5, maximum_rooting_gdd=1920,
                minimum_depth=0.30, maximum_depth=3.0, shape_factor=15)
    assert float(actual_rooting_depth_gdd(0, cumulative_gdd=0.0, **args)) == 0.0
    assert float(actual_rooting_depth_gdd(1, cumulative_gdd=0.0, **args)) == pytest.approx(0.30)
    assert float(actual_rooting_depth_gdd(100, cumulative_gdd=1920.0, **args)) == pytest.approx(3.0)
    assert float(actual_rooting_depth_gdd(181, cumulative_gdd=1920.0, **args)) == 0.0


def test_calendar_root_depth_matches_actual_rooting_depth_days_eager_jit_and_scan():
    # Direct cases from global.f90::ActualRootingDepthDays with RootPercentZmin=70.
    args = dict(cycle_days=125, emergence_days=5, maximum_rooting_days=100,
                minimum_depth=.30, maximum_depth=1.0, shape_factor=15)
    dap = jnp.arange(0, 127)

    def evaluate(day):
        return actual_rooting_depth_calendar(day, **args)

    def scanned(days):
        return jax.lax.scan(lambda carry, day: (carry, evaluate(day)), None, days)[1]

    expected = np.zeros(127)
    for day in range(1, 126):
        if day >= 100:
            expected[day] = 1.0
        elif day <= 2:
            expected[day] = .30
        else:
            shaped = .21 + .79 * np.exp((10. / 15.) *
                np.log((day - 2.) / (100. - 2.)))
            expected[day] = max(shaped, .30)
    for operation in (jax.vmap(evaluate), jax.jit(jax.vmap(evaluate)),
                      jax.jit(scanned)):
        np.testing.assert_allclose(operation(dap), expected, rtol=0., atol=1e-14)
    assert float(evaluate(6)) > .30
    assert float(evaluate(126)) == 0.0


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


def test_canopy_selects_relatively_wetter_topsoil_after_rain():
    selected = jax.jit(canopy_water_zone)
    common = ([.1, .1, .1], [0., 0., 0.], .3, 29., 13., 46., .1)
    top_wet = selected([.29, .15, .15], *common)
    root_wet = selected([.15, .29, .29], *common)
    assert bool(top_wet.topsoil_considered)
    assert float(top_wet.actual_mm) == pytest.approx(29.)
    assert float(top_wet.field_capacity_mm) == pytest.approx(29.)
    assert not bool(root_wet.topsoil_considered)
    assert float(root_wet.actual_mm) == pytest.approx(73.)
