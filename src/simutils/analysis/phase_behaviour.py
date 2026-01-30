import numpy as np
from simutils import ureg
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.results import ResultsGroup
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


TWO_PHASE = [
    [0.00, 0.29, 0.36, 0.37, 0.36, 0.34, 0.31, 0.25, 0.18, 0.11, 0.05, 0.00], # ethanol
    [1.00, 0.72, 0.55, 0.43, 0.33, 0.25, 0.19, 0.15, 0.12, 0.08, 0.05, 0.05], # water
    [0.00, 0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95], # octanol
]

def get_mass_density(ag, n_bins, range=None, dim="z"):
    """
    Calculates mass density across one dimension of the simulation box.
    
    Parameters
    ----------
    ag : AtomGroup
        AtomGroup for analysis.
    n_bins : int
        Number of bins.
    range : tuple, optional
        Minimum and maximum extent of the analysis, defaults to incude the whole axis.
    dim : str, optional
        Dimension to analyse (default = "z").

    Returns
    -------
    densities : ndarray
        Mass density values for each bin in g/cm^3.
    edges : ndarray
        Bin edges in Å.
    """
    dim_idx = {"x":0, "y":1, "z":2}
    d = dim_idx[dim]
    masses = ag.total_mass(compound="residues") # g mol-1
    positions = ag.center_of_mass(compound="residues")
    dimensions = ag.universe.dimensions[:3]
    if range is None:
        range = (0.0, dimensions[d])
    bin_mass, edges = np.histogram(
        positions[:, d],
        weights=masses,
        bins=n_bins,
        range=range
    )
    dimensions[d] = range[1] - range[0]
    volume = np.prod(dimensions) * ureg("Å^3")
    slice_volume = volume / n_bins
    bin_mass = bin_mass * ureg("g/mol")
    densities = (bin_mass / slice_volume)
    densities = densities / ureg("N_A")
    return densities.to("g/cm^3").magnitude, edges


def get_owe_composition(ag_all, ag_oct, ag_sol, ag_eth, n_bins, range=None, dim="z"):
    """
    Calculates OWE composition (wt %) across one dimension of the simulation box.
    
    Parameters
    ----------
    u : Universe
        Universe to analyse.
    ag_all : AtomGroup
        AtomGroup of whole system.
    ag_oct : AtomGroup
        Octanol AtomGroup.
    ag_sol : AtomGroup
        Water AtomGroup.
    ag_eth : AtomGroup
        Ethanol AtomGroup.
    n_bins : int
        Number of bins.
    range : tuple, optional
        Minimum and maximum extent of the analysis, defaults to incude the whole axis.
    dim : str, optional
        Dimension to analyse (default = "z").

    Returns
    -------
    bins : ndarray
        Bin centres.
    oct : ndarray
        Octanol composition.
    sol : ndarray
        Water composition.
    eth : ndarray
        Ethanol composition.
    """
    density, edges = get_mass_density(ag_all, n_bins, range=range, dim=dim)
    bins = 0.5 * (edges[:-1] + edges[1:])
    oct_density, _ = get_mass_density(ag_oct, n_bins, range=range, dim=dim)
    sol_density, _ = get_mass_density(ag_sol, n_bins, range=range, dim=dim)
    eth_density, _ = get_mass_density(ag_eth, n_bins, range=range, dim=dim)
    oct = 100.0 * oct_density / density
    sol = 100.0 * sol_density / density
    eth = 100.0 * eth_density / density
    return bins, edges, oct, sol, eth


class PhaseBehaviourThreshold(AnalysisBase):
    """
    Calculates the average composition of octanol-rich and water-rich phases of the OWE mixture. Phases assigned using octanol composition with significance threshold.

    Parameters
    ----------
    u : Universe
        Universe for phase behaviour analysis.
    composition : str
        Global O_W_E composition.
    n_bins : int, optional
        Number of bins along chosen axis (default = 50).
    threshold : float, optional
        Significance threshold (default = 0).
    range : tuple, optional
        Minimum and maximum extent of the analysis, defaults to incude the whole axis.
    dim : str, optional
        Dimension to analyse (default = "z").

    Attributes
    ----------
    results.edges : ndarray
        n_frame x (n_bins + 1) array of bin edges.
    results.oct : ndarray
        n_frame x n_bins array of octanol compositions.
    results.sol : ndarray
        n_frame x n_bins array of water compositions.
    results.eth : ndarray
        n_frame x n_bins array of ethanol compositions.
    results.labels : ndarray
        n_frame x n_bins array of phase assignment labels.
    results.composition : ndarray
        Compositions of each phase.
    """

    _analysis_algorithm_is_parallelizable = True

    @classmethod
    def get_supported_backends(cls):
        return ("serial", "multiprocessing", "dask")
    
    def __init__(
        self,
        u,
        composition,
        n_bins=50,
        threshold=0,
        range=None,
        dim="z",
    ):
        super(PhaseBehaviourThreshold, self).__init__(u.trajectory)
        self._u = u
        self._composition = composition
        self._n_bins = n_bins
        self._threshold = threshold
        self._range = range
        self._dim = dim

    def _prepare(self):
        self.results.edges = np.zeros((self.n_frames, self._n_bins + 1))
        self.results.oct = np.zeros((self.n_frames, self._n_bins))
        self.results.sol = np.zeros((self.n_frames, self._n_bins))
        self.results.eth = np.zeros((self.n_frames, self._n_bins))

        self._ag_all = self._u.atoms
        self._ag_oct = self._u.select_atoms("resname OCT")
        self._ag_sol = self._u.select_atoms("resname SOL or resname AFW")
        self._ag_eth = self._u.select_atoms("resname ETH")

    def _single_frame(self):
        _, edges, oct, sol, eth = get_owe_composition(
            self._ag_all,
            self._ag_oct,
            self._ag_sol,
            self._ag_eth,
            self._n_bins,
            range=self._range,
            dim=self._dim
        )
        
        self.results.edges[self._frame_index] = edges
        self.results.oct[self._frame_index] = oct
        self.results.sol[self._frame_index] = sol
        self.results.eth[self._frame_index] = eth

    def _get_aggregator(self):
        return ResultsGroup(
            lookup={
                "edges":ResultsGroup.ndarray_vstack,
                "oct":ResultsGroup.ndarray_vstack,
                "sol":ResultsGroup.ndarray_vstack,
                "eth":ResultsGroup.ndarray_vstack,
            }
        )

    def _conclude(self):
        o = float(self._composition.split("_")[0])
        mask_A = self.results.oct >= o + self._threshold
        mask_B = self.results.oct < o - self._threshold

        self.results.labels = np.full((self.n_frames, self._n_bins), np.nan)
        self.results.labels[mask_A] = 0
        self.results.labels[mask_B] = 1

        oct_A = self.results.oct[mask_A].mean()
        sol_A = self.results.sol[mask_A].mean()
        eth_A = self.results.eth[mask_A].mean()
        oct_B = self.results.oct[mask_B].mean()
        sol_B = self.results.sol[mask_B].mean()
        eth_B = self.results.eth[mask_B].mean()

        self.results.composition = [oct_A, oct_B, sol_A, sol_B, eth_A, eth_B]


class PhaseBehaviourKMeans(AnalysisBase):
    """
    Calculates the average composition of octanol-rich and water-rich phases of the OWE mixture. Phases assigned using K-means clustering.

    Parameters
    ----------
    u : Universe
        Universe for phase behaviour analysis.
    n_bins : int, optional
        Number of bins along chosen axis (default = 50).
    sil_threshold : float, optional
        Threshold silhouette score (default = 0.5).
    range : tuple, optional
        Minimum and maximum extent of the analysis, defaults to incude the whole axis.
    dim : str, optional
        Dimension to analyse (default = "z").

    Attributes
    ----------
    results.edges : ndarray
        n_frame x (n_bins + 1) array of bin edges.
    results.oct : ndarray
        n_frame x n_bins array of octanol compositions.
    results.sol : ndarray
        n_frame x n_bins array of water compositions.
    results.eth : ndarray
        n_frame x n_bins array of ethanol compositions.
    results.labels : ndarray
        n_frame x n_bins array of phase assignment labels.
    results.centers : ndarray
        Cluster centers.
    results.sil_score : float
        Average silhouette score for all sampled bins.
    results.composition : ndarray
        Compositions of each phase.
    """

    _analysis_algorithm_is_parallelizable = True

    @classmethod
    def get_supported_backends(cls):
        return ("serial", "multiprocessing", "dask")
    
    def __init__(
        self,
        u,
        n_bins=50,
        sil_threshold=0.5,
        range=None,
        dim="z",
    ):
        super(PhaseBehaviourKMeans, self).__init__(u.trajectory)
        self._u = u
        self._n_bins = n_bins
        self._sil_threshold = sil_threshold
        self._range = range
        self._dim = dim

    def _prepare(self):
        self.results.edges = np.zeros((self.n_frames, self._n_bins + 1))
        self.results.oct = np.zeros((self.n_frames, self._n_bins))
        self.results.sol = np.zeros((self.n_frames, self._n_bins))
        self.results.eth = np.zeros((self.n_frames, self._n_bins))

        self._ag_all = self._u.atoms
        self._ag_oct = self._u.select_atoms("resname OCT")
        self._ag_sol = self._u.select_atoms("resname SOL or resname AFW")
        self._ag_eth = self._u.select_atoms("resname ETH")

    def _single_frame(self):
        _, edges, oct, sol, eth = get_owe_composition(
            self._ag_all,
            self._ag_oct,
            self._ag_sol,
            self._ag_eth,
            self._n_bins,
            range=self._range,
            dim=self._dim
        )

        self.results.edges[self._frame_index] = edges
        self.results.oct[self._frame_index] = oct
        self.results.sol[self._frame_index] = sol
        self.results.eth[self._frame_index] = eth

    def _get_aggregator(self):
        return ResultsGroup(
            lookup={
                "edges":ResultsGroup.ndarray_vstack,
                "oct":ResultsGroup.ndarray_vstack,
                "sol":ResultsGroup.ndarray_vstack,
                "eth":ResultsGroup.ndarray_vstack,
            }
        )

    def _conclude(self):
        # K-means clustering based on octanol and water composition
        x = np.column_stack([self.results.oct.ravel(), self.results.sol.ravel(), self.results.eth.ravel()])
        init_centres = np.array([[100, 0, 0], [0, 100, 0]]) # cluster 0 = octanol-rich, cluster 1 = water-rich
        km = KMeans(n_clusters=2, init=init_centres).fit(x)

        self.results.labels = km.labels_.reshape((self.n_frames, self._n_bins))
        self.results.centers = km.cluster_centers_
        self.results.sil_score = silhouette_score(x, km.labels_)

        two_phases = self.results.sil_score > self._sil_threshold
        if two_phases:
            oct_A, sol_A, eth_A = self.results.centers[0]
            oct_B, sol_B, eth_B = self.results.centers[1]
            self.results.composition = np.array([oct_A, oct_B, sol_A, sol_B, eth_A, eth_B])
        else:
            self.results.composition = np.full(6, np.nan)
