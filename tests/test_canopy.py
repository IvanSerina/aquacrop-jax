import numpy as np
import pytest

from aquacrop_jax.canopy import (calculate_potential_et, canopy_at_time,
                                potential_canopy_gdd, temperature_stress)


def test_canopy_growth_switches_to_logistic_branch():
    assert float(canopy_at_time(0.0, 0.01, 0.1, 0.8)) == pytest.approx(0.01)
    assert 0.0 < float(canopy_at_time(50.0, 0.01, 0.1, 0.8)) <= 0.8


def test_gdd_canopy_guards_and_decline():
    args = (5.0, 100.0, 120.0, 0.01, 0.90, 0.04, 0.006)
    assert float(potential_canopy_gdd(2.0, *args)) == 0.0
    growing = float(potential_canopy_gdd(50.0, *args))
    declining = float(potential_canopy_gdd(110.0, *args))
    assert growing > 0.0
    assert declining >= 0.0
    assert float(potential_canopy_gdd(121.0, *args)) == 0.0


def test_temperature_stress_bounds():
    assert float(temperature_stress(0.0, 8.0, 0.0)) == 0.0
    assert float(temperature_stress(0.0, 8.0, 8.0)) == 1.0
    assert 0.0 < float(temperature_stress(0.0, 8.0, 4.0)) < 1.0


def test_potential_et_outside_crop_and_with_canopy():
    common = dict(
        emergence_day=2, full_canopy_day=30, senescence_day=90,
        harvest_day=120, eto_mm=5.0, maximum_crop_coefficient=1.15,
        cumulative_crop_coefficient_decline_pct=10.0,
        maximum_canopy_cover=0.9, maximum_withered_canopy=0.9,
        late_evaporation_effect_pct=60.0, co2_ppm=369.41,
        daily_gdd=10.0, minimum_gdd_full_transpiration=8.0)
    bare = calculate_potential_et(day_after_planting=0, canopy_cover=0.0,
                                  **common)
    crop = calculate_potential_et(day_after_planting=40, canopy_cover=0.8,
                                  **common)
    assert float(bare.transpiration_mm) == 0.0
    assert float(bare.evaporation_mm) == pytest.approx(5.5)
    assert float(crop.transpiration_mm) > 0.0
    assert float(crop.evaporation_mm) < float(bare.evaporation_mm)
