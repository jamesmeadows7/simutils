from os import PathLike
from string import Template


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