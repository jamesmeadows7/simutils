import numpy as np
import pytest

from simutils.analysis.cell import box_volume, calculate_cell_parameters


def test_calculate_cell_parameters_orthorhombic_box():
    H = np.diag([10.0, 20.0, 30.0])

    a, b, c, alpha, beta, gamma = calculate_cell_parameters(H)

    assert (a, b, c) == pytest.approx((10.0, 20.0, 30.0))
    assert (alpha, beta, gamma) == pytest.approx((90.0, 90.0, 90.0))


def test_calculate_cell_parameters_triclinic_box():
    H = np.array(
        [
            [10.0, 0.0, 0.0],
            [5.0, 5.0 * np.sqrt(3), 0.0],
            [0.0, 0.0, 10.0],
        ]
    )

    a, b, c, alpha, beta, gamma = calculate_cell_parameters(H)

    assert (a, b, c) == pytest.approx((10.0, 10.0, 10.0))
    assert (alpha, beta, gamma) == pytest.approx((90.0, 90.0, 60.0))


def test_box_volume_orthorhombic_box():
    H = np.diag([10.0, 20.0, 30.0])

    assert box_volume(H) == pytest.approx(6000.0)


def test_box_volume_triclinic_box():
    H = np.array(
        [
            [10.0, 0.0, 0.0],
            [5.0, 5.0 * np.sqrt(3), 0.0],
            [0.0, 0.0, 10.0],
        ]
    )

    assert box_volume(H) == pytest.approx(500.0 * np.sqrt(3))
