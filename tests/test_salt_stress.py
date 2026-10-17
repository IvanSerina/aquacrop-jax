import jax
import numpy as np
import pytest

from aquacrop_jax.salt_stress import run_salt_stress_average


def test_calibrated_salt_stress_mean_is_recurrent_and_masks_dormancy():
    operation = jax.jit(run_salt_stress_average)
    mean, reduction = operation([2., 9., 16., 16.], [True, True, False, True], 2., 16., (0., 1., 0.))
    assert np.asarray(mean) == pytest.approx([0., 25., 25., 50.])
    assert np.asarray(reduction) == pytest.approx([0., 25., 0., 50.])
    suffix = operation([16., 16.], [False, True], 2., 16., (0., 1., 0.), 2., 25.)
    np.testing.assert_allclose(suffix[0], mean[2:])
    np.testing.assert_allclose(suffix[1], reduction[2:])
