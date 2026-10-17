"""Full-precision operator comparisons using existing immutable Fortran objects."""
from pathlib import Path
import shutil
from ctypes.util import find_library
import subprocess

import jax
import numpy as np
import pytest

from aquacrop_jax.salinity import SaltState, concentrate_salts, initialize_salt, salt_geometry
from aquacrop_jax.salt_transport import salt_transport_step

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope='module')
def salt_probe(tmp_path_factory):
    reference = ROOT.parent / 'aquacrop-fortran/src'
    objects = [str(p) for p in reference.glob('*.o') if p.name != 'aquacrop.o']
    if not shutil.which('gcc') or not objects or not find_library('gfortran'):
        pytest.skip('Optional native probe requires gcc and prebuilt reference objects')
    probe = tmp_path_factory.mktemp('salt-probe') / 'probe'
    result = subprocess.run(['gcc', '-O2', '-no-pie', '-o', str(probe),
        str(ROOT / 'scripts/salt_probe.c'), *objects, '-l:libgfortran.so.5', '-lm'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return probe


def native_step(probe, state, theta, flux, thickness, ksat, irrigation, ec, gravel=0., solubility=100., concentrate=False):
    lines = [f'{len(theta)} {ksat} 46 29 {gravel} {solubility} 0 {irrigation} {ec}']
    for index in range(len(theta)):
        lines.append(f'{thickness[index]:.17g} {theta[index]:.17g} {flux[index]:.17g}')
        lines.extend(f'{float(salt):.17g} {float(deposit):.17g}' for salt, deposit in
                     zip(state.dissolved[index], state.deposited[index]))
    result = subprocess.run([str(probe)] + (["concentrate"] if concentrate else []), input='\n'.join(lines) + '\n',
                             text=True, capture_output=True, check=True)
    data = np.fromstring(result.stdout, sep=' ')
    stores = data[:-1].reshape(len(theta), 11, 2)
    return SaltState(stores[:, :, 0], stores[:, :, 1]), data[-1]


@pytest.mark.parametrize('ksat', [100., 500., 1200.])
def test_salt_transport_full_precision_native_parity(salt_probe, ksat):
    dz = np.asarray([.1, .15, .2])
    geometry = salt_geometry([46.] * 3, [29.] * 3, [ksat] * 3)
    theta = np.asarray([.29] * 3)
    state = initialize_salt([8., 4., 16.], theta, dz, [0.] * 3, 46., geometry)
    operation = jax.jit(salt_transport_step)
    # Multiple cell crossings, repeated wet/dry days and vertical diffusion.
    for updated, flux, irrigation, ec in [
        ([.46, .31, .29], [3., 0., 0.], 20., 5.),
        ([.387, .38, .33], [10., 8., 5.], 10., 2.),
        ([.29, .29, .29], [15., 10., 8.], 0., 0.),
        ([.14, .20, .25], [0., 0., 0.], 0., 0.),
    ]:
        expected, drain_ec = native_step(salt_probe, state, updated, flux, dz, ksat, irrigation, ec)
        result = operation(state, updated, flux, dz, [0.] * 3, 46., 29., geometry,
                           infiltrated_irrigation=irrigation, irrigation_ec=ec)
        np.testing.assert_allclose(result.state.dissolved, expected.dissolved, atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(result.state.deposited, expected.deposited, atol=1e-10, rtol=1e-12)
        assert float(result.drainage_ec) == pytest.approx(drain_ec, abs=1e-10)
        assert float(result.residual_g_m2) == pytest.approx(0., abs=1e-9)
        state = concentrate_salts(result.state, updated, dz, [0.] * 3, geometry)
        concentrated, _ = native_step(salt_probe, result.state, updated, [0.] * 3,
            dz, ksat, 0., 0., concentrate=True)
        np.testing.assert_allclose(state.dissolved, concentrated.dissolved, atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(state.deposited, concentrated.deposited, atol=1e-10, rtol=1e-12)


@pytest.mark.parametrize('shape', [0., 3., -2., 99.])
def test_salinity_and_ecsw_stress_match_native_functions(salt_probe, shape):
    from aquacrop_jax.salinity import adjusted_stomatal_stress, salinity_stress
    for ece, ecsw, wrel in [(0., 0., 0.), (4., 15., .4), (9., 25., .8), (17., 30., 1.)]:
        text = f'{ece} {ecsw} {ece * 46 / 29} {wrel} 2 16 100 {shape} 0.5 0.7 0.003 0.9 1'
        native = np.fromstring(subprocess.run([str(salt_probe), 'stress'], input=text,
            text=True, capture_output=True, check=True).stdout, sep=' ')
        actual = [float(salinity_stress(ece, 2., 16., shape)),
            float(adjusted_stomatal_stress(ece, ecsw, ece * 46 / 29, wrel,
                2., 16., 100., (.5, .7, .003), .9))]
        assert actual == pytest.approx(native, abs=1e-12)


@pytest.mark.parametrize('gdd', [False, True])
@pytest.mark.parametrize('red', [0, 5, 25, 75, 100])
def test_canopy_salt_parameters_match_native(salt_probe, gdd, red):
    from aquacrop_jax.salt_stress import salt_canopy_effect
    parameters = [red, 25, .05, .95, .17713, .012, 45, 55, 10, 180, 500, 650, 120, 1920, 0, int(gdd)]
    native = np.fromstring(subprocess.run([str(salt_probe), 'canopy'], input=' '.join(map(str, parameters)),
        text=True, capture_output=True, check=True).stdout, sep=' ')
    actual = salt_canopy_effect(*parameters)
    assert [float(v) for v in actual] == pytest.approx(native, abs=1e-12)


def test_precipitation_redissolution_and_gravel_match_native(salt_probe):
    dz, theta = np.asarray([.1, .15, .2]), np.asarray([.20, .25, .29])
    gravel, solubility, ksat = 20., 2., 100.
    geometry = salt_geometry([46.] * 3, [29.] * 3, [ksat] * 3)
    state = initialize_salt([20., 40., 10.], theta, dz, [gravel] * 3, 46., geometry, solubility)
    assert float(np.sum(state.deposited)) > 0.
    updated, flux = [.30, .33, .35], [5., 4., 2.]
    expected, drain_ec = native_step(salt_probe, state, updated, flux, dz, ksat,
                                    10., 5., gravel, solubility)
    result = jax.jit(salt_transport_step)(state, updated, flux, dz, [gravel] * 3,
        46., 29., geometry, infiltrated_irrigation=10., irrigation_ec=5., solubility=solubility)
    np.testing.assert_allclose(result.state.dissolved, expected.dissolved, atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(result.state.deposited, expected.deposited, atol=1e-10, rtol=1e-12)
    assert float(result.drainage_ec) == pytest.approx(drain_ec, abs=1e-12)
