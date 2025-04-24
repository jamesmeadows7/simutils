import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.lib.distances import capped_distance
from MDAnalysis.analysis.results import ResultsGroup

class RDF(AnalysisBase):
    """
    Calculates average radial distribution functions between the CoM of two groups of atoms.

    Parameters
    ----------
    ag1 : AtomGroup
        First AtomGroup.
    ag2 : AtomGroup
        Second AtomGroup.
    n_bins : int, optional
        Number of bins for the histogram (default = 75).
    range : tuple, optional
        Minimum and maximum extent of the radial distribution function in Å (default = (0.0, 15.0)).
    norm : str, optional
        Type of normalisation: rdf, density or none (default = rdf).
    exclude_same : bool, optional
        Exclude distances between same centre of masses (default = True).

    Attributes
    ----------
    results.bins : ndarray
        Histogram bin centres.
    results.edges : ndarray
        Histogram bin edges.
    results.rdf : ndarray
        Radial distribution function values.
    results.count : ndarray
        Raw histrogram counts.
    """

    _analysis_algorithm_is_parallelizable = True

    @classmethod
    def get_supported_backends(cls):
        return ("serial", "multiprocessing", "dask")
    
    def __init__(
        self,
        ag1,
        ag2,
        n_bins=75,
        range=(0.0, 15.0),
        norm="rdf",
        exclude_same=True,
    ):
        super(RDF, self).__init__(ag1.universe.trajectory)
        self._ag1 = ag1
        self._ag2 = ag2
        self._norm = str(norm).lower()
        self._rdf_settings = {"bins": n_bins, "range": range}
        self._exclude_same = exclude_same

    def _prepare(self):
        count, edges = np.histogram([-1], **self._rdf_settings) # empty histogram
        self.results.count = count
        self.results.edges = edges
        self.results.bins = 0.5 * (edges[:-1] + edges[1:])

        self._maxrange = self._rdf_settings["range"][1] # max range

    def _single_frame(self):
        ag1_coms = self._ag1.center_of_mass(unwrap=True, compound="residues")
        ag2_coms = self._ag2.center_of_mass(unwrap=True, compound="residues")

        pairs, distances = capped_distance(
            ag1_coms,
            ag2_coms,
            self._maxrange,
            box=self._ts.dimensions,
        )

        if self._exclude_same:
            # ignore distances between the same residues
            ag1_resindices = self._ag1.residues.resindices[pairs[:, 0]]
            ag2_resindices = self._ag2.residues.resindices[pairs[:, 1]]
            mask = np.where(ag1_resindices != ag2_resindices)[0]
            distances = distances[mask]

        count, _ = np.histogram(distances, **self._rdf_settings)
        self.results.count += count

        if self._norm == "rdf":
            self.results._volume = self._ts.volume

    def _get_aggregator(self):
        return ResultsGroup(
            lookup={
                "edges":ResultsGroup.ndarray_mean,
                "bins":ResultsGroup.ndarray_mean,
                "rdf":ResultsGroup.ndarray_sum,
                "count":ResultsGroup.ndarray_sum,
                "_volume":ResultsGroup.float_mean
            }
        )

    def _conclude(self):
        norm = self.n_frames
        if self._norm in ["rdf", "density"]:
            vols = np.power(self.results.edges, 3) # radial shell volumes
            norm *= 4 / 3 * np.pi * np.diff(vols)

        if self._norm == "rdf":
            N_1 = self._ag1.n_residues
            N_2 = self._ag2.n_residues
            N = N_1 * N_2 # should be N(N-1) for same atom groups

            # average number density
            box_vol = self.results._volume
            norm *= N / box_vol

        self.results.rdf = self.results.count / norm