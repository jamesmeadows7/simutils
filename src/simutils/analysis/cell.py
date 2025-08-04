import numpy as np
from MDAnalysis.analysis.base import AnalysisBase


class CellParameters(AnalysisBase):
    """
    Compute cell parameters.

    Parameters
    ----------
    u : Universe
        Universe for MSD analysis.
    corrected : bool, optional
        Account for GROMACS triclinic box constraints (default = true).

    Attributes
    ----------
    results.cell_matrices : ndarray
        Nx3x3 array of cell matrices. Components in Å.
    results.cell_parameters : ndarray
        Nx6 array of cell parameters: a, b, c, alpha, beta, gamma. Lengths in Å.
    results.avg_cell_matrix : ndarray
        3x3 array of average cell matrix values. Components in Å.
    results.avg_cell_parameters : ndarray
        Average cell parameters: a, b, c, alpha, beta, gamma. Lengths in Å.
    """

    def __init__(self, u, corrected=True):
        super(CellParameters, self).__init__(u.trajectory)
        self._u = u
        self._corrected = corrected

    def _prepare(self):
        self.results.cell_matrices = np.zeros((self.n_frames, 3, 3))

    def _single_frame(self):
        self.results.cell_matrices[self._frame_index] = self._ts.triclinic_dimensions

    def _conclude(self):
        if self._corrected:
            self.results.cell_matrices = continuous_cell_matrices(self.results.cell_matrices)

        self.results.cell_parameters = np.zeros((self.n_frames, 6))
        for i in range(self.n_frames):
            self.results.cell_parameters[i] = calculate_cell_parameters(self.results.cell_matrices[i])

        self.results.avg_cell_matrix = self.results.cell_matrices.mean(axis=0)
        if self._corrected:
            self.results.avg_cell_matrix = correct_cell_matrix(self.results.avg_cell_matrix)

        self.results.avg_cell_parameters = calculate_cell_parameters(self.results.avg_cell_matrix)


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
    beta  = np.degrees(np.arccos(np.dot(a_vec, c_vec) / (a * c)))
    gamma = np.degrees(np.arccos(np.dot(a_vec, b_vec) / (a * b)))
    return np.array([a, b, c, alpha, beta, gamma])


def continuous_cell_matrices(H_trajectory):
    """
    Convert a sequence of cell matrices to a continous representation suitable for averaging.

    Parameters
    ----------
    H_trajectory : ndarray
        Nx3x3 array containing N cell matrices.

    Returns
    -------
    continous H_trajectory : ndarray
        Nx3x3 array containing N cell matrices with continous representations.
    """
    continuous_H_trajectory = H_trajectory.copy()
    for i in range(1, len(H_trajectory)):
        H_prev = continuous_H_trajectory[i-1]
        H_curr = continuous_H_trajectory[i]
        # unwrap b_x
        shift = np.round((H_curr[1, 0] - H_prev[1, 0]) / H_curr[0, 0])
        H_curr[1] -= shift * H_curr[0]
        # unwrap c_x
        shift = np.round((H_curr[2, 0] - H_prev[2, 0]) / H_curr[0, 0])
        H_curr[2] -= shift * H_curr[0]
        # unwrap c_y
        shift = np.round((H_curr[2, 1] - H_prev[2, 1]) / H_curr[1, 1])
        H_curr[2] -= shift * H_curr[1]
        continuous_H_trajectory[i] = H_curr
    return continuous_H_trajectory


def correct_cell_matrix(H):
    """
    Ensure cell matrix conforms to GROMACS constraints.

    Parameters
    ----------
    H : ndarray
        3x3 cell matrix, rows correspond to box vectors a, b, c. H[1, 0] = b_x.

    Returns
    -------
    H_corrected : ndarray
        3x3 corrected cell matrix.
    """
    H_corrected = H.copy()
    assert H_corrected[0, 0] > 0.0 # a_x > 0
    assert H_corrected[1, 1] > 0.0 # b_y > 0
    assert H_corrected[2, 2] > 0.0 # c_z > 0
    assert H_corrected[0, 1] == 0.0 # a_y = 0
    assert H_corrected[0, 2] == 0.0 # a_z = 0
    assert H_corrected[1, 2] == 0.0 # b_z = 0
    H_corrected[1, 0] -= np.round(H_corrected[1, 0] / H_corrected[0, 0]) * H_corrected[0, 0] # reduce b_x
    H_corrected[2, 0] -= np.round(H_corrected[2, 0] / H_corrected[0, 0]) * H_corrected[0, 0] # reduce c_x
    H_corrected[2, 1] -= np.round(H_corrected[2, 1] / H_corrected[1, 1]) * H_corrected[1, 1] # reduce c_y
    return H_corrected