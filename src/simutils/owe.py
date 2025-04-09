import numpy as np
from simutils import ureg

MOLAR_MASS = [130.2314, 18.0154, 46.0694] * ureg("g/mol")
DENSITY = [824, 997, 789] * ureg("kg/m^3")
NATOMS = [27, 4, 9]


def get_n_owe(composition: list[float], box: list[float]) -> list[int]:
    """
    get_n_owe(composition, box)

    Returns the number of  molecules to achieve a target OWE wt% composition for a given box size.
    
    Parameters
    ----------
    composition : list
        The target OWE wt% composition [O, W, E].
    box : list
        The side lengths of the simulation box [a, b, c] in nm.

    Returns
    -------
    n_owe: list
        The number of octanol, water and ethanol molecules.
    """
    assert np.sum(composition) == 100
    mol_ratio = (composition / MOLAR_MASS) / np.sum(composition / MOLAR_MASS)  # normalised molar ratio
    mol_vol = MOLAR_MASS / (DENSITY * ureg("N_A"))  # molar volume
    vol_box = np.prod(box * ureg.nm)  # box volume
    n_owe = mol_ratio * (vol_box / np.sum(mol_ratio * mol_vol))
    return [int(i) for i in n_owe]