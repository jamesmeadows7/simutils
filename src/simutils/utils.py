from os import PathLike
from string import Template
from typing import Dict


def write_from_template(template_file: str | PathLike, output_file: str | PathLike, substitutions: Dict) -> None:
    with open(template_file, "r") as f:
        result = Template(f.read()).substitute(substitutions)
    with open(output_file, "w") as f:
        f.write(result)