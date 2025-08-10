from MDAnalysis.analysis.base import AnalysisBase
import numpy as np
import spglib as spg


Z = {"H":1, "C":6, "N":7, "O":8}

class Symmetry(AnalysisBase):
    """
    Compute symmetry. Non-parallelisable as it should be used in conjunction with ContinousCellMatrix transformation.

    Parameters
    ----------
    ag : AtomGroup
        AtomGroup for symmetry analysis.
    symprec : float, optional
        Symmetry seach tolerance in Å (default = 1e-1).

    Attributes
    ----------
    results.space_group_numbers : ndarray
        Array of space group numbers for each frame of the trajectory.
    """
    def __init__(self, ag, symprec=1e-1, verbose=True):
        super(Symmetry, self).__init__(ag.universe.trajectory, verbose=verbose)
        self._ag = ag
        self._symprec = symprec

    def _prepare(self):
        self.results.space_group_numbers = np.zeros(self.n_frames)

    def _single_frame(self):
        lattice = self._ts.triclinic_dimensions
        positions = np.linalg.solve(lattice.T, self._ag.atoms.positions.T).T
        numbers = [Z[name[0]] for name in self._ag.atoms.names]
        cell = (lattice, positions, numbers)
        symmetry_dataset = spg.get_symmetry_dataset(cell, symprec=self._symprec)
        self.results.space_group_numbers[self._frame_index] = symmetry_dataset.number

    def _conclude(self):
        pass