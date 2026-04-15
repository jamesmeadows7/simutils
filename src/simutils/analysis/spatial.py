from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.results import ResultsGroup
import numpy as np

class SpatialDistribution(AnalysisBase):

    _analysis_algorithm_is_parallelizable = True

    @classmethod
    def get_supported_backends(cls):
        return ("serial", "multiprocessing", "dask")
    
    def __init__(self, ag, n_bins):
        super(SpatialDistribution, self).__init__(ag.universe.trajectory)
        self._ag = ag
        self._n_bins = n_bins
        self._n_voxels = n_bins**3

    def _prepare(self):
        self.results.nmols = np.zeros((self.n_frames, self._n_voxels))

    def _single_frame(self):
        coms = self._ag.center_of_mass(unwrap=True, compound='residues')
        hist, _ = np.histogramdd(coms, bins=self._n_bins)
        self.results.nmols[self._frame_index] = hist.flatten()

    def _get_aggregator(self):
        return ResultsGroup(lookup={"nmols":ResultsGroup.ndarray_vstack})
    
    def _conclude(self):
        pass

        