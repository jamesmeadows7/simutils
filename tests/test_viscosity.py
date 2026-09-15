import numpy as np
import pytest

from simutils.analysis.viscosity import (
    autocorrelation,
    autocorrelation_fft,
    transform_pressure_tensor,
)


def test_transform_pressure_tensor_symmetrises_and_removes_trace():
    p_tensor = np.array([[[3.0, 1.0, 0.0], [3.0, 6.0, 0.0], [0.0, 0.0, 9.0]]])

    result = transform_pressure_tensor(p_tensor)

    expected = np.array([[[-3.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0]]])
    assert result == pytest.approx(expected)
    assert np.trace(result[0]) == pytest.approx(0.0)


def test_autocorrelation_methods_agree():
    p_ij = np.random.default_rng(0).normal(size=64)

    scipy_acf = autocorrelation(p_ij)
    fft_acf = autocorrelation_fft(p_ij)

    assert scipy_acf == pytest.approx(fft_acf)
