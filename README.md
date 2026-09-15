# simutils

A personal library of molecular dynamics simulation analysis and setup tools, built up over the course of my PhD in Computational Chemistry.

Most of the analysis code is built on top of the `Analysis Base` class provided by [MDAnalysis](https://www.mdanalysis.org/), incorporating additional libraries from the scientific Python ecosystem such as NumPy, SciPy, Pint and NetworkX.

## Analysis (`src/simutils/analysis/`)

### `ClusterAnalysis` — Molecular Cluster Detection and Structure

Molecular clustering, based on a centre of mass distance cutoff, using `self_capped_distance` from MDAnalysis and `networkx` to derive a graph of connected components.
`ClusterAnalysis` computes per-frame cluster size statistics and the cluster size distribution over a trajectory.
`ClusterStructure` computes an RDF-style distribution of different molecular components around a cluster.
`VisualiseClusters` writes a `.gro` file with clusters relabelled by resname for easy visualisation in VMD or Ovito.

```python
from simutils.analysis.clustering import ClusterAnalysis

clusters = ClusterAnalysis(ag, cutoff=5.0)
clusters.run()

clusters.results.avg_clust   # average cluster size per frame
clusters.results.size_dist   # average cluster size distribution
```

### `MSD` — Mean Squared Displacement by Centre of Mass

MDAnalysis's `EinsteinMSD` adapted to operate on residue centres of mass rather than individual atoms.
Applies the `NoJump` transformation before processing to unwrap the trajectory such that atoms never move more than half a periodic box length.
Added `compute_diff_coeffs` and `compute_diff_coefficent` methods to derive per-residue and molecule-averaged diffusion coefficients.

```python
from simutils.analysis.msd import MSD

msd = MSD(ag, msd_type="xyz", fft=True)
msd.run()

msd.results.timeseries  # molecule-averaged MSD vs lag-time
msd.compute_diff_coefficent(tau_min=100, tau_max=1000)  # diffusion coefficient, Å²/ps
```

### `RDF` — Radial Distribution Function by Centre of Mass

MDAnalysis's `InterRDF` adapted to compute the RDF between two groups using residue centres of mass rather than the positions of individual atoms.
Supports MDAnalysis's parallel analysis backends.

```python
from simutils.analysis.rdf import RDF

rdf = RDF(ag1, ag2, n_bins=100, range=(0.0, 20.0))
rdf.run(backend="multiprocessing", n_workers=4)

rdf.results.bins  # bin centres, Å
rdf.results.rdf   # g(r)
```

### `ContinousCellMatrix` and `CellParameters` — GROMACS Triclinic Cell Handling

`ContinousCellMatrix` is a trajectory transformation, built using `TransformationBase`, that undoes GROMACS's corrections to an overly-skewed triclinic cell, reversing discontinuous jumps in the cell matrix by comparing each frame to the previous one. `CellParameters` then properly time-averages cell lengths and angles from the continuous trajectory.

```python
from simutils.analysis.cell import ContinousCellMatrix, CellParameters

u.trajectory.add_transformations(ContinousCellMatrix())

cell = CellParameters(u)
cell.run()
cell.results.avg_cell_parameters  # a, b, c, alpha, beta, gamma
```

### Extending `HydrogenBondAnalysis`

Additional method `count_by_time_and_type` for MDAnalysis's `HydrogenBondAnalysis`.
Combines existing `count_by_time` and `count_by_type` methods into a single per-frame, per-type breakdown.

```python
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from simutils.analysis.hydrogenbonds import count_by_time_and_type

HydrogenBondAnalysis.count_by_time_and_type = count_by_time_and_type

hbonds = HydrogenBondAnalysis(u).run()
hbonds.count_by_time_and_type(norm="n_residues")
```

### Other Analysis Modules

- **`symmetry.py`** (`Symmetry`) — per-frame space-group detection via [spglib](https://spglib.readthedocs.io/), used alongside the `ContinousCellMatrix` transformation.
- **`viscosity.py`** — Green–Kubo viscosity from a GROMACS pressure tensor (`.edr`, via `panedr`), computed using [best practices](https://livecomsjournal.org/index.php/livecoms/article/view/v1i1e6324).
- **`blocking.py`** (`blocking_analysis`, `optimal_block`) — Flyvbjerg–Petersen block-averaging for automatic estimation of the statistical uncertainty of a correlated timeseries.
- **`phase_behaviour.py`** (`PhaseBehaviourThreshold`, `PhaseBehaviourKMeans`) — assigns octanol-rich/water-rich phases from mass density profiles (threshold or K-means), for a project-specific octanol/water/ethanol system.

## Setup (`src/simutils/setup/`)

GROMACS setup scripts for the octanol/water/ethanol/glycine project, calling the `gmx` binary.

- **`owe.py`** / **`owe2.py`** — atomistic system setup, for two glycine force-field variants.
- **`martini.py`** — coarse-grained system setup.
- **`cgf.py`** — shared force-field/polymorph/temperature constants.
- **`utils.py`** — template file generation and `gmx` subprocess helper functions.

## Plotting (`src/simutils/plotting/`)

- **`config.py`** (`set_plot_style`) — set a consistent matplotlib style, and a display-label lookup dictionary for force field names.
- **`timeseries.py`** (`plot_timeseries`, `plot_blocking_analysis`) — plotting helpers for timeseries (showing mean and SEM) and for a blocking analysis.
