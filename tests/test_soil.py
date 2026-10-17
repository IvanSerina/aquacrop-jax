from pathlib import Path

import pytest
import jax.numpy as jnp

from aquacrop_jax.soil import (discretize_profile, drainage_tau,
                              drainage_delta,
                              profile_water_mm, read_soil_profile,
                              resize_for_root_depth, theta_from_drainage,
                              vertical_drainage, water_balance_residual)

FIXTURE = Path(__file__).parents[2] / "aquacrop-fortran" / "testcase"


def test_ottawa_soil_profile_and_initial_compartments():
    soil = read_soil_profile(FIXTURE / "DATA" / "Ottawa.SOL")
    assert soil.curve_number == 46
    assert soil.readily_evaporable_water == 7
    assert len(soil.layers) == 1
    layer = soil.layers[0]
    assert (layer.thickness, layer.saturation, layer.field_capacity,
            layer.wilting_point, layer.ksat) == (1.5, 46.0, 29.0, 13.0, 1200.0)
    assert layer.gravel_volume == 0.0
    compartments = discretize_profile(soil)
    assert len(compartments.thickness) == 12
    assert compartments.thickness == pytest.approx((0.1,) * 12)
    assert compartments.layer_index == (0,) * 12


def test_ottawa_derived_hydraulic_values_are_float64():
    tau = drainage_tau(1200.0)
    water = profile_water_mm(jnp.full((12,), .29, jnp.float64),
                             jnp.full((12,), .1, jnp.float64),
                             jnp.zeros((12,), jnp.float64))
    assert tau.dtype == jnp.float64 and tau == pytest.approx(1.0)
    assert water.dtype == jnp.float64 and water == pytest.approx(348.0)


def test_ottawa_compartments_expand_exactly_like_fortran():
    soil = read_soil_profile(FIXTURE / "DATA" / "Ottawa.SOL")
    expanded = resize_for_root_depth(soil, discretize_profile(soil), 3.0)
    assert expanded.thickness == pytest.approx(
        (0.10, 0.15, 0.15, 0.20, 0.20, 0.25,
         0.25, 0.30, 0.30, 0.35, 0.35, 0.45))
    assert sum(expanded.thickness) == pytest.approx(3.05)
    assert expanded.layer_index == (0,) * 12
    water = profile_water_mm(
        jnp.full((12,), .29, jnp.float64),
        jnp.asarray(expanded.thickness, jnp.float64),
        jnp.zeros((12,), jnp.float64))
    assert water == pytest.approx(884.5)


def test_water_balance_closes_and_preserves_float64():
    # 100 + 10 rain = 102 end storage + 3 drain + 2 runoff + 1 E + 2 Tr
    residual = water_balance_residual(
        100, 0, 102, 0, 10, 0, 0, 0, 3, 2, 1, 2)
    assert residual.dtype == jnp.float64
    assert residual == pytest.approx(0.0)


def test_drainage_operators_match_and_invert_fortran_equations():
    delta = drainage_delta(0.35, 0.29, 46.0, 29.0, 1.0)
    restored = theta_from_drainage(delta, 0.29, 46.0, 29.0, 1.0)
    assert delta.dtype == jnp.float64
    assert 0.0 < delta <= 0.06
    assert restored == pytest.approx(0.35)


def test_drainage_respects_fc_saturation_and_zero_tau_guards():
    assert drainage_delta(0.29, 0.29, 46.0, 29.0, 1.0) == pytest.approx(0.0)
    assert drainage_delta(0.60, 0.29, 46.0, 29.0, 1.0) <= 0.1700001
    assert theta_from_drainage(0.01, 0.29, 46.0, 29.0, 0.0) == pytest.approx(0.56)


def test_vertical_drainage_flux_and_mass_conservation():
    theta0 = jnp.asarray([0.35, 0.35], jnp.float64)
    theta1, flux, bottom = vertical_drainage(
        theta0, [0.1, 0.1], [0.0, 0.0], [0.29, 0.29],
        46.0, 29.0, 1.0, 1200.0)
    assert flux[1] == pytest.approx(2.0 * flux[0])
    initial = profile_water_mm(theta0, [0.1, 0.1], [0.0, 0.0])
    final = profile_water_mm(theta1, [0.1, 0.1], [0.0, 0.0])
    assert initial - final == pytest.approx(bottom)


def test_vertical_drainage_applies_ksat_limit_without_losing_water():
    theta0 = jnp.asarray([0.35, 0.35], jnp.float64)
    theta1, flux, bottom = vertical_drainage(
        theta0, [0.1, 0.1], [0.0, 0.0], [0.29, 0.29],
        46.0, 29.0, 1.0, 1.0)
    assert flux == pytest.approx([1.0, 1.0])
    initial = profile_water_mm(theta0, [0.1, 0.1], [0.0, 0.0])
    final = profile_water_mm(theta1, [0.1, 0.1], [0.0, 0.0])
    assert bottom == pytest.approx(1.0)
    assert initial - final == pytest.approx(1.0)
