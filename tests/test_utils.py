from pathlib import Path

import pytest

from simutils.setup.utils import write_from_template

TEMPLATE = Path(__file__).parent / "data" / "md.mdp"


def test_write_from_template_substitutes_mdp_temperature(tmp_path):
    output = tmp_path / "output.mdp"

    write_from_template(TEMPLATE, output, {"temp": 298.15})

    result = output.read_text()
    assert "${temp}" not in result
    assert result.count("298.15") == 2


def test_write_from_template_missing_substitution_raises_key_error(tmp_path):
    output = tmp_path / "output.mdp"

    with pytest.raises(KeyError):
        write_from_template(TEMPLATE, output, {})
