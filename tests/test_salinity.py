import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aquacrop_jax.salinity import (SaltState, active_cells,
    adjusted_stomatal_stress, compartment_conductivity, concentrate_salts,
    initialize_salt, root_zone_conductivity, round_fortran, salt_geometry,
    salinity_stress, solution_deposit)
from aquacrop_jax.salt_transport import mix_cells, salt_transport_step


def test_geometry_and_mobility_fortran_equations():
    g = salt_geometry([46., 46., 46.], [29.] * 3, [100., 500., 1200.])
    assert np.asarray(g.count).tolist() == [11, 4, 2]
    assert np.asarray(g.width) == pytest.approx([.46/12, .46/5, .46/3])
    assert np.asarray(g.mobility[:, 0]) == pytest.approx([
        (1000.**(1/7)-1)/999., (1000.**(1/3)-1)/999., 1.])
    assert np.asarray(round_fortran([-.5, .5, 1.5])) == pytest.approx([0, 0, 2])
    assert np.asarray(active_cells([g.width[0], g.upper[1], .46], g)).tolist() == [0, 2, 1]


def test_solution_deposit_dissolves_again_and_conserves_mass():
    dry = solution_deposit(1., 250., 0.)
    assert dry == (100., 150.)
    wet = solution_deposit(3., *dry)
    assert wet == (250., 0.)


def test_conductivity_counts_deposits_and_partial_root_compartment():
    g = salt_geometry([46.] * 2, [29.] * 2, [100.] * 2)
    state = initialize_salt([2., 8.], [.29, .29], [.1, .2], [0., 0.], [46.] * 2, g)
    ec = compartment_conductivity(state, [.29, .29], [.1, .2], [0., 0.], 46., 29.)
    assert np.asarray(ec.ece) == pytest.approx([2., 8.])
    zone = root_zone_conductivity(ec, [.1, .2], .2)
    assert float(zone.ece) == pytest.approx(5.)
    assert float(zone.ecsw) == pytest.approx(5 * 46 / 29)
    concentrated = concentrate_salts(state, [.01, .01], [.1, .2], [0., 0.], g)
    assert float(jnp.sum(concentrated.dissolved + concentrated.deposited)) == pytest.approx(
        float(jnp.sum(state.dissolved + state.deposited)))
    assert np.all(np.asarray(concentrated.dissolved[:, 1:]) == 0.)


def test_stress_response_and_ecsw_extra_stress():
    assert np.asarray(salinity_stress([1., 2., 9., 16., 20.], 2., 16.)) == pytest.approx([1., 1., .5, 0., 0.])
    assert float(salinity_stress(9., 2., 16., 3.)) == pytest.approx(.8175744761936437)
    assert float(salinity_stress(9., 2., 16., 99.)) == pytest.approx(.49)
    adjusted = jax.jit(adjusted_stomatal_stress)(4., 12., 6., .5, 2., 16., 100., (0., 1., 0.), .9)
    assert 0. < float(adjusted) < .9
    assert float(adjusted_stomatal_stress(4., 12., 6., 0., 2., 16., 100., (0., 1., 0.), .9)) == .9


def test_cell_mixing_independent_mass_and_solubility_oracle():
    first, second = mix_cells(.2, 10., 20., 64., 256., 0., 0.)
    # EC 10 and 20 mix toward 16 2/3 with mobility .2.
    assert float(first.dissolved) == pytest.approx(72.53333333333333)
    assert float(second.dissolved) == pytest.approx(247.46666666666667)
    assert float(first.dissolved + second.dissolved) == pytest.approx(320.)


def test_irrigation_transport_leaching_jit_and_closed_salt_budget():
    geometry = salt_geometry([46.] * 2, [29.] * 2, [1200.] * 2)
    state = SaltState(jnp.zeros((2, 11)), jnp.zeros((2, 11)))
    operation = jax.jit(salt_transport_step)
    result = operation(state, [.29, .29], [20., 20.], [.1, .1], [0., 0.],
        46., 29., geometry, infiltrated_irrigation=20., irrigation_ec=5.)
    assert float(result.infiltrated_g_m2) == pytest.approx(64.)
    assert 0. < float(result.drained_g_m2) < 64.
    assert float(result.residual_g_m2) == pytest.approx(0., abs=1e-12)
    assert np.all(np.asarray(result.state.dissolved) >= -1e-12)
    dry = operation(result.state, [.29, .29], [0., 0.], [.1, .1], [0., 0.],
        46., 29., geometry)
    assert float(dry.drained_g_m2) == 0.
    assert float(dry.residual_g_m2) == pytest.approx(0., abs=1e-12)
