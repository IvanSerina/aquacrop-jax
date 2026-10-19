"""The protected-seedling part of native Ottawa canopy development."""
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from aquacrop_jax.canopy import (ProtectedSeedlingState, calculate_potential_et,
                                protected_seedling_gdd_step)
from aquacrop_jax.fertility import soil_fertility_effect
from aquacrop_jax.kernels import degrees_day
from aquacrop_jax.outputs import read_daily_output


TESTCASE = Path(__file__).parents[2] / 'aquacrop-fortran/testcase'


def test_protected_seedling_matches_first_three_native_ottawa_days():
    daily = read_daily_output(TESTCASE / 'OUTP_REF/OttawaPRMday.OUT')
    native = [row.mapping() for row in daily.records if row.run == 1][:3]
    assert [row['canopy_pct'] for row in native] == [5.4, 5.9, 6.4]
    temperature = np.asarray([(row['tmin'], row['tmax']) for row in native])
    gdd = degrees_day(5., 30., temperature[:, 0], temperature[:, 1], 3)
    cumulative = jnp.cumsum(gdd)
    effect = soil_fertility_effect(50., 2.35, .79, -.16, 6.26)
    assert (float(effect.reduction_growth_pct),
            float(effect.reduction_cover_pct)) == (24., 40.)
    # EffectSoilFertilitySalinityStress bypasses fertility while
    # VirtualTimeCC < DaysToGermination: Ottawa's first virtual day is 0 < 1.
    red_growth = jnp.asarray([0., 24., 24.])
    red_cover = jnp.asarray([0., 40., 40.])

    def step(state, forcing):
        total, daily_gdd, growth, cover = forcing
        return protected_seedling_gdd_step(
            state, total, daily_gdd, 5., 1920., 1920.,
            .05, .95, .012, .006, growth, cover)

    initial = ProtectedSeedlingState(jnp.float64(0.), jnp.bool_(True))
    # Record every state, including the day-3 protected-seedling transition.
    def scan_step(state, forcing):
        next_state, valid = step(state, forcing)
        return next_state, (next_state.cover, next_state.protected, valid)

    forcing = cumulative, gdd, red_growth, red_cover
    eager_state, eager = lax.scan(scan_step, initial, forcing)
    jit_state, compiled = jax.jit(lambda f: lax.scan(scan_step, initial, f))(forcing)
    for state, (cover, protected, valid) in ((eager_state, eager),
                                              (jit_state, compiled)):
        np.testing.assert_array_equal(valid, [True, True, True])
        np.testing.assert_array_equal(protected, [True, True, False])
        assert not bool(state.protected)
        # day.OUT prints one decimal percentage point, so half its unit is
        # the tightest possible comparison with this native fixture.
        np.testing.assert_allclose(100. * cover,
                                   [row['canopy_pct'] for row in native],
                                   rtol=0., atol=.050000000001)
        assert cover.dtype == jnp.float64
        # BUDGET_module calculates potential ET immediately after canopy.
        # The GDD-mode ET DAP is the rounded-GDD calendar conversion, not
        # necessarily the printed DAP (Ottawa day 3 converts to day 4).
        for i, row in enumerate(native):
            demand = calculate_potential_et(
                [1, 2, 4][i], 1, 39, 180, 180, cover[i], row['eto'],
                1.15, 11., .95, jnp.max(cover[:i + 1]), 60.,
                row['co2'], gdd[i], 8.)
            assert abs(float(demand.evaporation_mm) -
                       row['potential_evaporation']) <= .050000000001
            assert abs(float(demand.transpiration_mm) -
                       row['potential_transpiration']) <= .050000000001


def test_protected_seedling_guard_and_phase_boundary_are_explicit():
    initial = ProtectedSeedlingState(jnp.float64(0.), jnp.bool_(True))
    before, valid = protected_seedling_gdd_step(
        initial, 4., 4., 5., 100., 120., .05, .95, .012, .006, 0., 0.)
    assert bool(valid) and float(before.cover) == 0.
    after, valid = protected_seedling_gdd_step(
        ProtectedSeedlingState(jnp.float64(.064), jnp.bool_(False)),
        30., 10., 5., 100., 120., .05, .95, .012, .006, 24., 40.)
    assert not bool(valid)
    assert float(after.cover) == .064
