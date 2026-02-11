import numpy as np
import string
import networkx as nx
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.results import ResultsGroup
from MDAnalysis.lib.distances import capped_distance, self_capped_distance, apply_PBC
from MDAnalysis import Writer

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
    results.edges : ndarray
        Histogram bin edges.
    results.bins : ndarray
        Histogram bin centres.
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
        _, edges = np.histogram([-1], bins=self._n_bins, range=(0, self._n_residues)) # empty histogram
        self.results.edges = edges
        self.results.bins = 0.5 * (edges[:-1] + edges[1:])
        self.results.size_dist = np.zeros(self._n_bins)

    def _single_frame(self):
        coms = self._ag.center_of_mass(unwrap=True, compound="residues")
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
            lookup={
                "n_clust":ResultsGroup.ndarray_hstack,
                "avg_clust":ResultsGroup.ndarray_hstack,
                "w_avg_clust":ResultsGroup.ndarray_hstack,
                "max_clust":ResultsGroup.ndarray_hstack,
                "edges":ResultsGroup.ndarray_mean,
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
            self.results.edges /= self._n_residues
            self.results.bins /= self._n_residues
        self.results.size_dist /= (self.n_frames * self._n_residues)


class ClusterStructure(AnalysisBase):
    """
    Structure of clusters.
    """

    _analysis_algorithm_is_parallelizable = True

    @classmethod
    def get_supported_backends(cls):
        return ("serial", "multiprocessing", "dask")
    
    def __init__(
            self,
            ag1,
            ag2,
            cutoff,
            cluster_size_limits,
            n_bins=75,
            range=(0.0, 15.0),
        ):
        super(ClusterStructure, self).__init__(ag1.universe.trajectory)
        self._ag1 = ag1
        self._ag2 = ag2
        self._cutoff = cutoff
        self._rdf_settings = {"bins": n_bins, "range": range}
        self._cluster_size_limits = cluster_size_limits

    def _prepare(self):
        count, edges = np.histogram([-1], **self._rdf_settings) # empty histogram
        self.results.count = count
        self.results.edges = edges
        self.results.bins = 0.5 * (edges[:-1] + edges[1:])

        self._maxrange = self._rdf_settings["range"][1] # max range

        self.results._volume = 0.0
        self.results._n_clusters = 0
        self.results._valid_frames = 0

    def _single_frame(self):
        # find clusters of ag1
        ag1_coms = self._ag1.center_of_mass(unwrap=True, compound="residues")
        pairs = self_capped_distance(ag1_coms, self._cutoff, box=self._ts.dimensions, return_distances=False)
        G = nx.Graph()
        G.add_nodes_from(range(self._ag1.n_residues))
        G.add_edges_from(pairs)
        clusters = list(nx.connected_components(G))

        clusters = [cluster for cluster in clusters if (len(cluster) > self._cluster_size_limits[0] and len(cluster) < self._cluster_size_limits[1])]

        if len(clusters) == 0:
            return

        # compute cluster CoMs
        cluster_coms = []
        for cluster in clusters:
            residues = self._ag1.residues[list(cluster)]
            atoms = residues.atoms
            com = atoms.center_of_mass()
            # com = cluster_com_pbc(residues, self._ts.dimensions)
            cluster_coms.append(com)
        cluster_coms = np.array(cluster_coms)

        # compute distance between cluster CoMs and ag2 CoMs
        ag2_coms = self._ag2.center_of_mass(unwrap=True, compound="residues")
        pairs, distances = capped_distance(
            cluster_coms,
            ag2_coms,
            self._maxrange,
            box=self._ts.dimensions,
        )

        count, _ = np.histogram(distances, **self._rdf_settings)
        self.results.count += count

        self.results._volume += self._ts.volume
        self.results._n_clusters += len(clusters)
        self.results._valid_frames += 1

    def _get_aggregator(self):
        return ResultsGroup(
            lookup={
                "edges":ResultsGroup.ndarray_mean,
                "bins":ResultsGroup.ndarray_mean,
                "rdf":ResultsGroup.ndarray_sum,
                "count":ResultsGroup.ndarray_sum,
                "_volume":ResultsGroup.ndarray_sum,
                "_n_clusters":ResultsGroup.ndarray_sum,
                "_valid_frames":ResultsGroup.ndarray_sum,
            }
        )

    def _conclude(self):
        vols = 4 / 3 * np.pi * np.diff(np.power(self.results.edges, 3) )# radial shell volumes

        rho_local = self.results.count / (self.results._n_clusters * vols)
        rho_bulk = self._ag2.n_residues / (self.results._volume / self.results._valid_frames)

        self.results.rdf = rho_local / rho_bulk


class ClusterStructureSelf(AnalysisBase):
    """
    Structure of clusters.
    """

    _analysis_algorithm_is_parallelizable = True

    @classmethod
    def get_supported_backends(cls):
        return ("serial", "multiprocessing", "dask")
    
    def __init__(
            self,
            ag,
            cutoff,
            cluster_size_limits,
            n_bins=75,
            range=(0.0, 15.0),
        ):
        super(ClusterStructureSelf, self).__init__(ag.universe.trajectory)
        self._ag = ag
        self._cutoff = cutoff
        self._rdf_settings = {"bins": n_bins, "range": range}
        self._cluster_size_limits = cluster_size_limits

    def _prepare(self):
        count, edges = np.histogram([-1], **self._rdf_settings) # empty histogram
        self.results.count = count
        self.results.edges = edges
        self.results.bins = 0.5 * (edges[:-1] + edges[1:])

        self._maxrange = self._rdf_settings["range"][1] # max range

        self.results._volume = 0.0
        self.results._n_clusters = 0
        self.results._valid_frames = 0

    def _single_frame(self):
        # find clusters of ag
        ag_coms = self._ag.center_of_mass(unwrap=True, compound="residues")
        pairs = self_capped_distance(ag_coms, self._cutoff, box=self._ts.dimensions, return_distances=False)
        G = nx.Graph()
        G.add_nodes_from(range(self._ag.n_residues))
        G.add_edges_from(pairs)
        clusters = list(nx.connected_components(G))

        clusters = [cluster for cluster in clusters if (len(cluster) > self._cluster_size_limits[0] and len(cluster) < self._cluster_size_limits[1])]

        if len(clusters) == 0:
            return

        # compute cluster CoMs
        for cluster in clusters:
            residues = self._ag.residues[list(cluster)]
            atoms = residues.atoms
            com = atoms.center_of_mass()
            # com = cluster_com_pbc(residues, self._ts.dimensions)

            # compute distance between cluster CoMs and cluster molecule CoMs
            mol_coms = residues.center_of_mass(unwrap=True, compound="residues")
            pairs, distances = capped_distance(
                com,
                mol_coms,
                self._maxrange,
                box=self._ts.dimensions,
            )

            count, _ = np.histogram(distances, **self._rdf_settings)
            self.results.count += count

        self.results._volume += self._ts.volume
        self.results._n_clusters += len(clusters)
        self.results._valid_frames += 1

    def _get_aggregator(self):
        return ResultsGroup(
            lookup={
                "edges":ResultsGroup.ndarray_mean,
                "bins":ResultsGroup.ndarray_mean,
                "rdf":ResultsGroup.ndarray_sum,
                "count":ResultsGroup.ndarray_sum,
                "_volume":ResultsGroup.ndarray_sum,
                "_n_clusters":ResultsGroup.ndarray_sum,
                "_valid_frames":ResultsGroup.ndarray_sum,
            }
        )

    def _conclude(self):
        vols = 4 / 3 * np.pi * np.diff(np.power(self.results.edges, 3) )# radial shell volumes

        rho_local = self.results.count / (self.results._n_clusters * vols)
        rho_bulk = self._ag.n_residues / (self.results._volume / self.results._valid_frames)

        self.results.rdf = rho_local / rho_bulk

class VisualiseClusters(AnalysisBase):
    """
    Create gro file.
    """
    
    def __init__(self, ag, cutoff):
        super(VisualiseClusters, self).__init__(ag.universe.trajectory)
        self._ag = ag
        self._cutoff = cutoff
        self._n_residues = ag.n_residues

    def _prepare(self):
        self._writer = Writer("clusters.gro", self._ag.n_atoms, multiframe=False)

    def _single_frame(self):
        coms = self._ag.center_of_mass(unwrap=True, compound="residues")
        pairs = self_capped_distance(coms, self._cutoff, box=self._ts.dimensions, return_distances=False)
        G = nx.Graph()
        G.add_nodes_from(range(self._n_residues))
        G.add_edges_from(pairs)
        clusters = list(nx.connected_components(G))
        sorted_clusters = sorted(clusters, key=len, reverse=True)

        res_to_cluster = {}
        for cluster_number, cluster in enumerate(sorted_clusters):
            for residue in cluster:
                res_to_cluster[residue + 1] = cluster_number + 1
        new_resnames = [
            cluster_to_resname(res_to_cluster[i]) if i in res_to_cluster else "UNK"
            for i in self._ag.residues.resids
        ]
        self._ag.residues.resnames = new_resnames

        self._writer.write(self._ag.atoms)

    def _conclude(self):
        self._writer.close()


def cluster_com_pbc(residues, box):
    """
    Compute PBC-aware cluster CoM.
    """
    coms = residues.center_of_mass(compound="residues")
    masses = np.array([res.atoms.masses.sum() for res in residues])
    ref = coms[0] # reference molecule
    disp = coms - ref
    disp = apply_PBC(disp, box)
    unwrapped = ref + disp
    total_mass = masses.sum()
    com = np.sum(unwrapped * masses[:, None], axis=0) / total_mass
    com = apply_PBC(com.reshape(1, 3), box)[0]
    return com


def cluster_to_resname(cluster_number, repeat=3):
    letter = string.ascii_uppercase[cluster_number - 1]
    return letter * repeat