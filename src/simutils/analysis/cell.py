"""Tools for computing average cell dimensions, allowing for GROMACS skew corrections."""

import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.transformations.base import TransformationBase


class ContinousCellMatrix(TransformationBase):
    """
    Undo GROMACS cell corrections for proper averaging.

    GROMACS may add or subtract box vectors if the simulation cell becomes too skewed.
    This produces discontinuous jumps in the triclinic cell matrix between frames,
    which prevents a proper time-average of the cell parameters. This transformation
    compares each frame's cell matrix to the previous frame's and reverses any such
    jump, giving a continuous cell matrix over the trajectory.
    """

    def __init__(self, max_threads=None):
        super().__init__(max_threads=max_threads, parallelizable=False)
        self._H_prev = None

    def _transform(self, ts):
        H = ts.triclinic_dimensions

        if self._H_prev is not None:
            # unwrap b_x
            shift = np.round((H[1, 0] - self._H_prev[1, 0]) / H[0, 0])
            H[1] -= shift * H[0]
            # unwrap c_x
            shift = np.round((H[2, 0] - self._H_prev[2, 0]) / H[0, 0])
            H[2] -= shift * H[0]
            # unwrap c_y
            shift = np.round((H[2, 1] - self._H_prev[2, 1]) / H[1, 1])
            H[2] -= shift * H[1]

        ts.triclinic_dimensions = H.copy()
        self._H_prev = ts.triclinic_dimensions
        return ts


class CellParameters(AnalysisBase):
    """
    Compute cell parameters.

    Parameters
    ----------
    u : Universe
        Universe for cell parameter analysis.
    tau_min : float, optional
        Start time for averaging in ps (default = first frame time).
    tau_max : float, optional
        End time for averaging in ps (default = last frame time).

    Attributes
    ----------
    results.cell_matrices : ndarray
        Nx3x3 array of cell matrices. Components in Å.
    results.cell_parameters : ndarray
        Nx6 array of cell parameters: a, b, c, alpha, beta, gamma. Lengths in Å. Angles in degrees.
    results.avg_cell_matrix : ndarray
        3x3 array of average cell matrix values. Components in Å.
    results.avg_cell_parameters : ndarray
        Cell parameters of the average cell matrix. Lengths in Å. Angles in degrees.
    """

    def __init__(self, u, t_min=None, t_max=None):
        super(CellParameters, self).__init__(u.trajectory)
        self._u = u
        self._t_min = t_min
        self._t_max = t_max

    def _prepare(self):
        self.results.cell_matrices = np.zeros((self.n_frames, 3, 3))

    def _single_frame(self):
        self.results.cell_matrices[self._frame_index] = self._ts.triclinic_dimensions

    def _conclude(self):
        self.results.cell_parameters = np.zeros((self.n_frames, 6))
        for i in range(self.n_frames):
            self.results.cell_parameters[i] = calculate_cell_parameters(
                self.results.cell_matrices[i]
            )

        times = self.times

        if self._t_min is None:
            self._t_min = times[0]
        if self._t_max is None:
            self._t_max = times[-1]

        mask = (times >= self._t_min) & (times <= self._t_max)

        self.results.avg_cell_matrix = self.results.cell_matrices[mask].mean(axis=0)
        self.results.avg_cell_parameters = calculate_cell_parameters(
            self.results.avg_cell_matrix
        )


def calculate_cell_parameters(H):
    """
    Calculate cell lengths and angles from cell matrix.

    Parameters
    ----------
    H : ndarray
        3x3 cell matrix, rows correspond to box vectors a, b, c. H[1, 0] = b_x.

    Returns
    -------
    cell_parameters : ndarray
        Cell parameters a, b, c, alpha, beta, gamma.
    """
    a_vec, b_vec, c_vec = H[0], H[1], H[2]
    a = np.linalg.norm(a_vec)
    b = np.linalg.norm(b_vec)
    c = np.linalg.norm(c_vec)
    alpha = np.degrees(np.arccos(np.dot(b_vec, c_vec) / (b * c)))
    beta = np.degrees(np.arccos(np.dot(a_vec, c_vec) / (a * c)))
    gamma = np.degrees(np.arccos(np.dot(a_vec, b_vec) / (a * b)))
    return np.array([a, b, c, alpha, beta, gamma])


def box_volume(H):
    """
    Calculate box volume from cell matrix.

    Parameters
    ----------
    H : ndarray
        3x3 cell matrix, rows correspond to box vectors a, b, c. H[1, 0] = b_x.

    Returns
    -------
    box_volume : float
        Box volume.
    """
    return np.abs(np.linalg.det(H))
