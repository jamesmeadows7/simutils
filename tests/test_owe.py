from simutils.setup.owe import (
    get_conc_gly,
    get_n_gly,
    get_system_size,
)


def test_get_system_size():
    n_mols, n_atoms = get_system_size(n_owe=[10, 20, 5], n_gly=3)

    assert n_mols == 38
    assert n_atoms == 10 * 27 + 20 * 4 + 5 * 9 + 3 * 10


def test_get_n_gly_and_get_conc_gly_inverse():
    box = [4.0, 4.0, 4.0]  # nm

    conc = get_conc_gly(100, box)

    assert get_n_gly(conc, box) == 100
