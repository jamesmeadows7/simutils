"""Extra analysis helper for MDAnalysis's HydrogenBondAnalysis."""

import numpy as np


def count_by_time_and_type(self, norm="None"):
    """
    Count hydrogen bonds per frame, grouped by (donor, acceptor) residue-name pair.

    Call by monkey-patching onto a HydrogenBondAnalysis instance that has been run().

    Parameters
    ----------
    self : HydrogenBondAnalysis
        A HydrogenBondAnalysis instance that has already been run().
    norm : str, optional
        Type of normalisation: None, total_hbonds or n_residues (default = None).

    Returns
    -------
    dict
        Mapping of (donor_resname, acceptor_resname) tuples to arrays of per-frame counts.

    Notes
    -----
    If HydrogenBondAnalysis is constructed with a `between` selection restricting
    donor/acceptor residues, TIP4P (and other 4-site) water models will resolve the
    acceptor to the MW virtual site rather than the OW oxygen.
    """
    frames = self.results.hbonds[:, 0]
    frames = (
        frames - self.start
    ) / self.step  # convert frame numbers to index in self.frames
    d = self.u.atoms[self.results.hbonds[:, 1].astype(np.intp)]
    a = self.u.atoms[self.results.hbonds[:, 3].astype(np.intp)]
    hbond_data = np.array([frames.astype(int), d.resnames, a.resnames]).T

    pair_counts = {}

    for frame, d_resname, a_resname in hbond_data:
        pair = tuple(sorted((d_resname, a_resname)))
        if pair not in pair_counts:
            pair_counts[pair] = np.zeros_like(self.frames)
        pair_counts[pair][frame] += 1

    if norm == "total_hbonds":
        counts = self.count_by_time()
        pair_counts = {key: item / counts for key, item in pair_counts.items()}

    if norm == "n_residues":
        norm_pair_counts = {}
        for key, item in pair_counts.items():
            d_resname, a_resname = key
            n_d = self.u.select_atoms(f"resname {d_resname}").n_residues
            n_a = self.u.select_atoms(f"resname {a_resname}").n_residues
            if d_resname == a_resname:
                norm_pair_counts[key] = item / (n_d * (n_d - 1) / 2)
            else:
                norm_pair_counts[key] = item / (n_d * n_a / 2)
        pair_counts = norm_pair_counts

    return dict(sorted(pair_counts.items()))
