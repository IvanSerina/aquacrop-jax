import numpy as np
import pytest

from aquacrop_jax.production import (co2_water_productivity_factor,
                                    reference_harvest_index,
                                    run_production_trajectory)


def test_co2_reference_is_neutral_and_c4_is_unadjusted():
    assert float(co2_water_productivity_factor(369.41, 15.0, 100)) == pytest.approx(1.0)
    assert float(co2_water_productivity_factor(700.0, 40.0, 100)) == pytest.approx(1.0)
    assert float(co2_water_productivity_factor(550.0, 15.0, 100)) > 1.0


def test_reference_harvest_index_guards_and_limit():
    assert float(reference_harvest_index(9, 10, 50, 1.0)) == 0.0
    assert 0.0 <= float(reference_harvest_index(30, 10, 50, 1.0)) <= 50.0
    assert float(reference_harvest_index(100, 10, 50, 1.0)) == pytest.approx(50.0)


def test_production_recurrence_accumulates_biomass_and_yield():
    result = run_production_trajectory(
        [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [5.0, 5.0, 5.0],
        [369.41, 369.41, 369.41], [10.0, 11.0, 12.0], 15.0, 100,
        flowering_day=0, maximum_hi_pct=100, daily_hi_rate_pct=100,
        reproductive_crop=False, grain_crop=False)
    assert np.asarray(result.biomass) == pytest.approx([0.06, 0.15, 0.27])
    assert np.asarray(result.unlimited_biomass) == pytest.approx([0.09, 0.21, 0.36])
    assert np.asarray(result.yield_mass) == pytest.approx(np.asarray(result.biomass))
