from datetime import date
from pathlib import Path

import numpy as np
import pytest

from aquacrop_jax.climate import date_to_day_number
from aquacrop_jax.legacy_inputs import (co2_for_period, groundwater_forcing,
    initial_conditions, manual_irrigation_forcing, field_fertility_stress)


def test_field_fertility_stress_matches_native_ottawa_management():
    path = Path(__file__).parents[2] / 'aquacrop-fortran/testcase/DATA/Ottawa.MAN'
    assert field_fertility_stress(path) == 50
    assert field_fertility_stress(None) == 0


def test_manual_irr_preserves_fortran_ec_lookahead_and_absolute_days(tmp_path):
    path = tmp_path / 'manual.IRR'
    path.write_text('manual\n7.3\n1\n100\n1\n100\n\nDay Depth EC\n1 20 2\n3 10 8\n')
    depth, ec = manual_irrigation_forcing(path, [100, 101, 102, 103], 90)
    assert depth == pytest.approx([20., 0., 10., 0.])
    assert ec == pytest.approx([8., 0., 8., 0.])


def test_groundwater_interpolates_then_rounds_depth_to_centimetres(tmp_path):
    path = tmp_path / 'table.GWT'
    path.write_text('table\n7.3\n2\n1\n1\n2014\n\nDay Depth EC\n===\n1 1.0 2\n4 2.0 5\n')
    first = date_to_day_number(date(2014, 1, 1))
    depth, ec = groundwater_forcing(path, np.arange(first, first + 5))
    assert depth == pytest.approx([1., 1.33, 1.67, 2., 2.])
    assert ec == pytest.approx([2., 3., 4., 5., 5.])


def test_initial_layers_weight_partial_compartments_and_extend_last_layer(tmp_path):
    path = tmp_path / 'salt.SW0'
    path.write_text('salt\n7.3\n-9\n0\n-9\n0\n0\n0\n2\n\nThickness WC ECe\n===\n.15 20 2\n.1 40 6\n')
    state = initial_conditions(path, [.1, .1, .2], 29., 46.)
    assert state.theta == pytest.approx([.2, .3, .4])
    assert state.ece == pytest.approx([2., 4., 6.])


def test_co2_uses_legacy_average_of_endpoint_years():
    path = Path(__file__).parents[2] / 'aquacrop-fortran/testcase/SIMUL/MaunaLoa.CO2'
    first = date_to_day_number(date(2014, 5, 21))
    last = date_to_day_number(date(2014, 10, 31))
    assert co2_for_period(path, first, last) == 398.82
