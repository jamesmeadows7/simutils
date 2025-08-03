import numpy as np

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
        H = continuous_H_trajectory[i]
        # unwrap b_x
        shift = np.round((H[1, 0] - H_prev[1, 0]) / H[0, 0])
        H[1, 0] -= shift * H[0, 0] 
        # unwrap c_x
        shift = np.round((H[2, 0] - H_prev[2, 0]) / H[0, 0])
        H[2, 0] -= shift * H[0, 0] 
        # unwrap c_y
        shift = np.round((H[2, 1] - H_prev[2, 1]) / H[1, 1])
        H[2, 1] -= shift * H[1, 1] 
        continuous_H_trajectory[i] = H
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