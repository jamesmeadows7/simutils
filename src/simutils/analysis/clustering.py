import numpy as np
import networkx as nx
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.results import ResultsGroup
from MDAnalysis.lib.distances import self_capped_distance

class ClusterAnalysis(AnalysisBase):
    """
    Cluster analysis based on the CoM distance between residues.

    Parameters
    ----------
    ag : AtomGroup
        AtomGroup for cluster analysis.
    cutoff : float
        Distance cutoff in Å.
    norm : bool, optional
        Normalise results by total number of residues (default = True).
    n_bins : int, optional
        Number of bins for the cluster size histogram (default = 50).

    Attributes
    ----------
    results.n_clust : ndarray
        Number of clusters per frame.
    results.avg_clust : ndarray
        Average cluster size per frame.
    results.w_avg_clust : ndarray
        Weighted average cluster size per frame.
    results.max_clust : ndarray
        Size of the largest cluster per frame.
    results.bins : ndarray
        Size distribution histogram bins.
    results.size_dist : ndarray
        Average cluster size distribution.
    """

    _analysis_algorithm_is_parallelizable = True

    @classmethod
    def get_supported_backends(cls):
        return ("serial", "multiprocessing", "dask")
    
    def __init__(self, ag, cutoff, norm=True, n_bins=50):
        super(ClusterAnalysis, self).__init__(ag.universe.trajectory)
        self._ag = ag
        self._cutoff = cutoff
        self._norm = norm
        self._n_bins = n_bins
        self._n_residues = ag.n_residues

    def _prepare(self):
        self.results.n_clust = np.zeros(self.n_frames)
        self.results.avg_clust = np.zeros(self.n_frames)
        self.results.w_avg_clust = np.zeros(self.n_frames)
        self.results.max_clust = np.zeros(self.n_frames)
        _, bins = np.histogram([-1], bins=self._n_bins, range=(0, self._n_residues)) # empty histogram
        self.results.bins = bins
        self.results.size_dist = np.zeros(self._n_bins)

    def _single_frame(self):
        coms = self._ag.center_of_mass(unwrap=True, compound='residues')
        pairs = self_capped_distance(coms, self._cutoff, box=self._ts.dimensions, return_distances=False)
        G = nx.Graph()
        G.add_nodes_from(range(self._n_residues))
        G.add_edges_from(pairs)
        clusters = list(nx.connected_components(G))
        cluster_sizes = [len(c) for c in clusters]
        self.results.n_clust[self._frame_index] = len(cluster_sizes)
        self.results.avg_clust[self._frame_index] = np.mean(cluster_sizes)
        self.results.w_avg_clust[self._frame_index] = np.sum(np.square(cluster_sizes))/np.sum(cluster_sizes)
        self.results.max_clust[self._frame_index] = np.max(cluster_sizes)
        counts, _ = np.histogram(cluster_sizes, bins=self._n_bins, range=(0, self._n_residues), weights=cluster_sizes)
        self.results.size_dist += counts

    def _get_aggregator(self):
        return ResultsGroup(
            lookup={"n_clust":ResultsGroup.ndarray_hstack,
                    "avg_clust":ResultsGroup.ndarray_hstack,
                    "w_avg_clust":ResultsGroup.ndarray_hstack,
                    "max_clust":ResultsGroup.ndarray_hstack,
                    "bins":ResultsGroup.ndarray_mean,
                    "size_dist":ResultsGroup.ndarray_sum,
            }
        )
    
    def _conclude(self):
        if self._norm:
            self.results.n_clust /= self._n_residues
            self.results.avg_clust /= self._n_residues
            self.results.w_avg_clust /= self._n_residues
            self.results.max_clust /= self._n_residues
            self.results.bins /= self._n_residues
        self.results.size_dist /= (self.n_frames * self._n_residues)