import numpy as np
from tqdm import tqdm
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.transformations.nojump import NoJump
from scipy.stats import linregress

class MSD(AnalysisBase):
    """
    Mean squared displacement based on molecules CoMs.

    Parameters
    ----------
    ag : AtomGroup
        AtomGroup for MSD analysis.
    msd_type : str, optional
        Dimensions to be included in the MSD (default = xyz).
    fft : bool, optional
        Use FFT based algorithm (default = True).

    Attributes
    ----------
    dim_fac : int
        Dimensionality of the MSD.
    results.timeseries : ndarray
        Averaged MSD over all the particles with respect to lag-time.
    results.msds_by_particle : ndarray
        MSD of each individual particle with respect to lag-time.
    """

    def __init__(self, ag, msd_type="xyz", fft=True):
        # apply NoJump transformation
        ag.universe.trajectory.add_transformations(NoJump())
        super(MSD, self).__init__(ag.universe.trajectory)
        self._ag = ag
        self._msd_type = msd_type
        self._parse_msd_type()
        self._fft = fft
        self._n_residues = ag.n_residues

    def _parse_msd_type(self):
        keys = {
            "x": [0],
            "y": [1],
            "z": [2],
            "xy": [0, 1],
            "xz": [0, 2],
            "yz": [1, 2],
            "xyz": [0, 1, 2],
        }
        self._msd_type = self._msd_type.lower()
        try:
            self._dim = keys[self._msd_type]
        except KeyError:
            raise ValueError("Invalid msd type.")
        self.dim_fac = len(self._dim)

    def _prepare(self):
        self.results.msds_by_particle = np.zeros((self.n_frames, self._n_residues))
        self.results._position_array = np.zeros((self.n_frames, self._n_residues, self.dim_fac))

    def _single_frame(self):
        # don't need to unwrap coms because we use NoJump
        coms = self._ag.center_of_mass(unwrap=False, compound="residues")
        self.results._position_array[self._frame_index] = coms[:, self._dim]

    def _conclude(self):
        if self._fft:
            self._conclude_fft()
        else:
            self._conclude_simple()

    def _conclude_simple(self):
        lagtimes = np.arange(1, self.n_frames)
        positions = self.results._position_array.astype(np.float64)
        for lag in tqdm(lagtimes):
            disp = positions[:-lag, :, :] - positions[lag:, :, :]
            sqdist = np.square(disp).sum(axis=-1)
            self.results.msds_by_particle[lag, :] = np.mean(sqdist, axis=0)
        self.results.timeseries = self.results.msds_by_particle.mean(axis=1)

    def _conclude_fft(self):
        try:
            import tidynamics
        except ImportError:
            raise ImportError("tidynamics not found")

        positions = self.results._position_array.astype(np.float64)
        for n in tqdm(range(self._n_residues)):
            self.results.msds_by_particle[:, n] = tidynamics.msd(positions[:, n, :])
        self.results.timeseries = self.results.msds_by_particle.mean(axis=1)

    def compute_diff_coeffs(self, tau_min=None, tau_max=None):
        """
        Compute diffusion coefficient for each indvidual residue.

        Parameters
        ----------
        tau_min : float, optional
            Start time for linear fit in ps (default = first frame time).
        tau_max : float, optional
            End time for linear fit in ps (default = last frame time).

        Attributes
        ----------
        diff_coeffs : ndarray
            Diffusion coefficients for each residue in Å²/ps.
        """
        times = self.times

        if tau_min is None:
            tau_min = times[0]
        if tau_max is None:
            tau_max = times[-1]

        mask = (times >= tau_min) & (times <= tau_max)

        diff_coeffs = np.zeros(self._n_residues)
        for i in range(self._n_residues):
            lin = linregress(times[mask], self.results.msds_by_particle[mask, i])
            diff_coeffs[i] = lin.slope / (2*self.dim_fac)
        return diff_coeffs
    
    def compute_diff_coefficent(self, tau_min=None, tau_max=None):
        """
        Compute diffusion coefficient from the molecule-averaged MSD.

        Parameters
        ----------
        tau_min : float, optional
            Start time for linear fit in ps (default = first frame time).
        tau_max : float, optional
            End time for linear fit in ps (default = last frame time).

        Attributes
        ----------
        diff_coeff : float
            Diffusion coefficient in Å²/ps.
        """
        times = self.times

        if tau_min is None:
            tau_min = times[0]
        if tau_max is None:
            tau_max = times[-1]

        mask = (times >= tau_min) & (times <= tau_max)
        lin = linregress(times[mask], self.results.timeseries[mask])
        diff_coeff = lin.slope / (2*self.dim_fac)
        return diff_coeff