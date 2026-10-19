"""Unrounded native parity for the fertility full-canopy timing limiter."""
from ctypes.util import find_library
import json
from pathlib import Path
import shutil
import subprocess

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pytest

from aquacrop_jax.fertility import time_to_max_canopy_sf


ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / 'tests/fixtures/fertility_canopy_fortran73.json').read_text())['cases']
INPUTS = jnp.asarray([case['input'] for case in CASES], jnp.float64)
EXPECTED = np.asarray([case['expected'] for case in CASES], np.int32)


def _matrix(result):
    return np.column_stack(result)


def test_time_to_max_canopy_sf_matches_native_eager_jit_and_scan():
    eager = np.asarray([tuple(int(v) for v in time_to_max_canopy_sf(*case['input']))
                        for case in CASES])
    np.testing.assert_array_equal(eager, EXPECTED)

    def scan_body(carry, row):
        return carry, time_to_max_canopy_sf(*row)

    compiled = jax.jit(lambda rows: lax.scan(scan_body, (), rows)[1])
    np.testing.assert_array_equal(_matrix(compiled(INPUTS)), EXPECTED)
    np.testing.assert_array_equal(_matrix(jax.jit(jax.vmap(
        lambda row: time_to_max_canopy_sf(*row)))(INPUTS)), EXPECTED)


def test_fertility_canopy_fixture_matches_live_native_objects(tmp_path):
    reference = ROOT.parent / 'aquacrop-fortran/src'
    objects = sorted(p for p in reference.glob('*.o') if p.name != 'aquacrop.o')
    if not shutil.which('gcc') or not objects or not find_library('gfortran'):
        pytest.skip('Optional native probe requires gcc and prebuilt reference objects')
    probe = tmp_path / 'fertility-canopy-probe'
    result = subprocess.run(['gcc', '-O2', '-no-pie', '-o', str(probe),
        str(ROOT / 'scripts/fertility_canopy_probe.c'), *map(str, objects),
        '-l:libgfortran.so.5', '-lm'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    text = str(len(CASES)) + '\n' + '\n'.join(
        ' '.join(str(value) for value in case['input']) for case in CASES) + '\n'
    actual = subprocess.run([str(probe)], input=text, text=True,
                            capture_output=True, check=True)
    values = np.asarray([list(map(int, line.split())) for line in actual.stdout.splitlines()])
    np.testing.assert_array_equal(values, EXPECTED)
