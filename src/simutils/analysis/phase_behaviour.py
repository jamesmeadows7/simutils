import numpy as np
from simutils import ureg

def get_mass_density(ag, n_bins, range=False, dim="z"):
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
    if not range:
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


def get_owe_composition(u, n_bins, range=False, dim="z"):
    """
    Calculates OWE composition (wt %) across one dimension of the simulation box.
    
    Parameters
    ----------
    u : Universe
        Universe to analyse.
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
    ag = u.atoms
    density, edges = get_mass_density(ag, n_bins, range=range, dim=dim)
    bins = 0.5 * (edges[:-1] + edges[1:])
    ag = u.select_atoms("resname OCT")
    oct_density, _ = get_mass_density(ag, n_bins, range=range, dim=dim)
    ag = u.select_atoms("resname OCT")
    oct_density, _ = get_mass_density(ag, n_bins, range=range, dim=dim)
    ag = u.select_atoms("resname SOL or resname AFW")
    sol_density, _ = get_mass_density(ag, n_bins, range=range, dim=dim)
    ag = u.select_atoms("resname ETH")
    eth_density, _ = get_mass_density(ag, n_bins, range=range, dim=dim)
    oct = 100.0 * oct_density / density
    sol = 100.0 * sol_density / density
    eth = 100.0 * eth_density / density
    return bins, oct, sol, eth