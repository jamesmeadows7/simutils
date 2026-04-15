import numpy as np
from MDAnalysis.analysis.base import AnalysisBase, Results


class SpatialDensity(AnalysisBase):

    def __init__(self, ag, n_bins):
        super(SpatialDensity, self).__init__(ag.universe.trajectory)
        self._ag = ag
        self._n_bins = n_bins

    def _prepare(self):
        self.results.density_grid = np.zeros(self._n_bins)

    def _single_frame(self):
        coms = self._ag.center_of_mass(wrap=True, unwrap=True, compound="residues")

        H, edges = np.histogramdd(
            coms,
            bins=self._n_bins,
            range=tuple((0,l) for l in self._ts.dimensions[:3]),
            weights=None,
        )

        ########### wrap ###############

        print(np.sum(H))

        self.results.density_grid += H

    def _conclude(self):
        self.results.density_grid / self.n_frames