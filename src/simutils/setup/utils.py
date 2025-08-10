from os import PathLike
from string import Template
import subprocess


def write_from_template(template: PathLike, output: PathLike, substitutions: dict) -> None:
    """
    Replaces $-identifiers in template file with values in substitution dictionary and writes to output file.

    Parameters
    __________
    template : PathLike
        Path to the template file.
    output : PathLike
        Path to the output file.
    substitutions : dict
        Dictionary of substitutions with keys corresponding to $-identifiers in template file.
    """
    with open(template, "r") as f:
        result = Template(f.read()).substitute(substitutions)
    with open(output, "w") as f:
        f.write(result)


def replicate_unitcell(unitcell: PathLike, supercell: PathLike, n_replicas: list) -> None:
    """
    Replaces $-identifiers in template file with values in substitution dictionary and writes to output file.

    Parameters
    __________
    unitcell : PathLike
        Path of the gro file containing the unitcell.
    supercell : PathLike
        Path of the created supercell.
    n_replicas : list
        Number of replicas in each dimension [nx, ny, nz].
    """
    nx, ny, nz = n_replicas
    p = subprocess.run(
    f"gmx genconf -f {unitcell} -nbox {nx} {ny} {nz} -o {supercell}",
    shell=True,
    check=True,
    capture_output=True,
    )