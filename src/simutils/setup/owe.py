import subprocess, re, logging, shutil
import numpy as np
from os import PathLike
from simutils import ureg
from simutils.setup.utils import write_from_template

logger = logging.getLogger(__name__)

OWE_MOLAR_MASS = [130.2314, 18.0154, 46.0694] * ureg("g/mol")
OWE_DENSITY = [824, 997, 789] * ureg("kg/m^3")
OWE_N_ATOMS = [27, 4, 9]

GLY_MOLAR_MASS = 75.0675 * ureg("g/mol")
GLY_DENSITY = 1610 * ureg("kg/m^3")
GLY_N_ATOMS = 10


def get_n_owe(composition: list[float], box: list[float], n_gly: int) -> list[int]:
    """
    Calculates the number of molecules required to achieve a target OWE wt% composition for a given box size.
    
    Parameters
    ----------
    composition : list
        Target OWE wt% composition [O, W, E].
    box : list
        Side lengths of the simulation box [a, b, c] in nm.
    n_gly : int
        Number of glycine molecules.

    Returns
    -------
    n_owe : list
        Number of octanol, water and ethanol molecules.
    """
    assert np.sum(composition) == 100
    owe_mol_ratio = (composition / OWE_MOLAR_MASS) / np.sum(composition / OWE_MOLAR_MASS)  # OWE normalised molar ratio
    owe_vol = OWE_MOLAR_MASS / (OWE_DENSITY * ureg("N_A"))  # OWE volume per molecule
    gly_vol = GLY_MOLAR_MASS / (GLY_DENSITY * ureg("N_A")) # glycine volume per molecule
    box_vol = np.prod(box * ureg.nm)  # box volume
    rem_vol = box_vol - gly_vol*n_gly # remaining volume
    n_owe = owe_mol_ratio * (rem_vol / np.sum(owe_mol_ratio * owe_vol))
    return [int(i) for i in n_owe]


def get_system_size(n_owe:list, n_gly:int) -> tuple[int,int]:
    """
    Calculates the number of molecules and number of atoms to be insered into the simulation box.
    
    Parameters
    ----------
    n_owe : list
        Number of octanol, water and ethanol molecules.
    n_gly : int
        Number of glycine molecules.

    Returns
    -------
    n_mols : int
        Number of molecules in the simulation box.
    n_atoms : int
        Number of atoms in the simulation box.
    """
    n_mols = np.sum(n_owe) + n_gly
    n_atoms = np.sum(np.multiply(n_owe, OWE_N_ATOMS)) + (n_gly * GLY_N_ATOMS)
    return n_mols, n_atoms


def create_empty_box(box_gro: PathLike, title: str, box: list[float]) -> None:
    """
    Creates an empty gro file with a given box size to which molecules can be subsequently inserted.

    Parameters
    ----------
    box_gro : PathLike
        Path of the gro file to be written.
    title : str
        Title of the gro file.
    box : list
        Side lengths of the simulation box [a, b, c] in nm.
    """
    logger.info(f"Creating empty box with side lengths {box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f} nm at {box_gro}.")
    with open(box_gro, "w") as f:
        f.write(f"{title}\n")
        f.write(f"{0:5}\n")
        f.write(f"{box[0]:10.5f}{box[1]:10.5f}{box[2]:10.5f}")


def insert_molecules(box_gro: PathLike, mol_gro: PathLike, n_mol: int) -> None:
    """
    Attempts to insert molecules into the simulation box. 

    Parameters
    ----------
    box_gro : str or PathLike
        Path of the gro file to be written to.
    mol_gro : str or PathLike
        Path of the gro file containing molecule to insert.
    n_mol : int
        Number of molecules to insert.
    """
    p = subprocess.run(
        f"gmx insert-molecules -f {box_gro} -ci {mol_gro} -nmol {n_mol} -o {box_gro}",
        shell=True,
        check=True,
        capture_output=True
    )
    match = re.search(r"Added (\d+) molecules", p.stderr.decode())
    n_ins = int(match.group(1))
    assert n_ins == n_mol


def solvate(box_gro: PathLike, sol_gro: PathLike, n_sol: int) -> None:
    """
    Attempts to insert solvent molecules into the simulation box. 

    Parameters
    ----------
    box_gro : str or PathLike
        Path of the gro file to be written to.
    sol_gro : str or PathLike
        Path of the gro file containing solvent to insert.
    n_sol : int
        Number of solvent molecules to insert.
    """
    p = subprocess.run(
    f"gmx solvate -cp {box_gro} -cs {sol_gro} -maxsol {n_sol} -o {box_gro}",
    shell=True,
    check=True,
    capture_output=True
    )
    match = re.search(r"Generated solvent containing \d+ atoms in (\d+) residues", p.stderr.decode())
    n_ins = int(match.group(1))
    assert n_ins == n_sol


def create_simulation_box(sim_dir: PathLike, template_dir: PathLike, title: str, n_owe: list, n_gly: int, box: list, scale: float) -> None:
    """
    Creates simulation box containing octanol, water, ethanol and glycine molecules.

    Parameters
    ----------
    sim_dir : PathLike
        Path of the simulation directory.
    template_dir : str or PathLike
        Path of the template directory where the molecule gro files are stored.
    title : int
        Title of the gro file.
    n_owe : list
        Number of octanol, water and ethanol molecules.
    n_gly : int
        Number of glycine molecules.
    box : list
        Side lengths of the simulation box [a, b, c] in nm.
    scale : float
        Factor by which to scale the box side lengths to ensure all molecules can be inserted. 
    """
    scaled_box = [scale*i for i in box]
    create_empty_box(sim_dir / "box.gro", title, scaled_box)
    n_oct, n_wat, n_eth = n_owe
    if n_gly != 0:
        logger.info(f"Attempting to insert {n_gly} glycine molecules.")
        insert_molecules(sim_dir / "box.gro", template_dir / "glycine.gro", n_gly)
    if n_oct != 0:
        logger.info(f"Attempting to insert {n_oct} octanol molecules.")
        insert_molecules(sim_dir / "box.gro", template_dir / "octanol.gro", n_oct)
    if n_eth != 0:
        logger.info(f"Attempting to insert {n_eth} ethanol molecules.")
        insert_molecules(sim_dir / "box.gro", template_dir / "ethanol.gro", n_eth)
    if n_wat != 0:
        logger.info(f"Attempting to insert {n_wat} water molecules.")
        solvate(sim_dir / "box.gro", template_dir / "tip4p2005.gro", n_wat)
    logger.info("All requested molecules successfully inserted. Removing backup files.")
    [file.unlink() for file in sim_dir.glob("#box.gro.*#")]


def write_topol_file(sim_dir: PathLike, template_dir: PathLike, title: str, n_owe: list, n_gly: int):
    """
    Writes topol.top file accounting for the number of each molecule in the simulation box.

    Parameters
    ----------
    sim_dir : PathLike
        Path of the simulation directory.
    template_dir : str or PathLike
        Path of the template directory where the molecule gro files are stored.
    title : int
        Title of the gro file.
    n_owe : list
        Number of octanol, water and ethanol molecules.
    n_gly : int
        Number of glycine molecules.
    """
    includes = ""
    molecules = ""
    n_oct, n_wat, n_eth = n_owe
    if n_gly != 0:
        includes += "\n#include \"opls/glycine/opls2020_cm1a.itp\""
        molecules += f"{"GCN":^13}{n_gly:^9}\n"
    if n_oct != 0:
        includes += "\n#include \"opls/octanol/opls2020.itp\""
        molecules += f"{"OCT":^13}{n_oct:^9}\n"
    if n_eth != 0:
        includes += "\n#include \"opls/ethanol/opls2020.itp\""
        molecules += f"{"ETH":^13}{n_eth:^9}\n"
    if n_wat != 0:
        includes += "\n#include \"opls/water/tip4p2005.itp\""
        molecules += f"{"SOL":^13}{n_wat:^9}\n"
    write_from_template(template_dir / "topol.top", sim_dir / "topol.top", {"includes":includes, "title":title, "molecules":molecules})


def setup_owe_simulation(sim_dir: PathLike, template_dir: PathLike, title: str, n_owe: list, n_gly: int, box: list, scale: float = 1.1) -> None:
    """
    Sets up OWE simulation directory with gro file, topol file and mdp files.

    Parameters
    ----------
    sim_dir : PathLike
        Path of the simulation directory.
    template_dir : str or PathLike
        Path of the template directory where the molecule gro files are stored.
    title : int
        Title of the gro file.
    n_owe : list
        Number of octanol, water and ethanol molecules.
    n_gly : int
        Number of glycine molecules.
    box : list
        Side lengths of the simulation box [a, b, c] in nm.
    scale : float
        Factor by which to scale the box side lengths to ensure all molecules can be inserted (default = 1.1). 
    """
    sim_dir.mkdir(parents=True, exist_ok=True)
    n_mols, n_atoms = get_system_size(n_owe, n_gly)
    logger.info(f"Creating system with {n_mols} molecules and {n_atoms} atoms.")
    create_simulation_box(sim_dir, template_dir, title, n_owe, n_gly, box, scale)
    logger.info(f"Copying mdp files.")
    shutil.copy(template_dir / "min.mdp", sim_dir / "min.mdp")
    shutil.copy(template_dir / "nvt.mdp", sim_dir / "nvt.mdp")
    shutil.copy(template_dir / "npt.mdp", sim_dir / "npt.mdp")
    shutil.copy(template_dir / "md.mdp", sim_dir / "md.mdp")
    logger.info(f"Writing topol file.")
    write_topol_file(sim_dir, template_dir, title, n_owe, n_gly)
    logger.info(f"Writing slurm file.")
    write_from_template(template_dir / "run.sh", sim_dir / "run.sh", {"jobname":sim_dir.name})


def get_n_gly(conc_gly: float, box: list[float]) -> int:
    """
    Calculates the number of glycine molecules required to achieve a target glycine concentration for a given box size.
    
    Parameters
    ----------
    conc_gly : float
        Target glycine concentration in mg/cm^3.
    box : list
        Side lengths of the simulation box [a, b, c] in nm.

    Returns
    -------
    n_gly : int
        Number of glycine molecules.
    """
    conc_gly = conc_gly * ureg("mg/cm^3")
    box_vol = np.prod(box * ureg.nm)
    n_gly = conc_gly * box_vol * ureg("N_A") / GLY_MOLAR_MASS
    return int(n_gly.to_base_units())


def get_conc_gly(n_gly: int, box: list[float]) -> int:
    """
    Calculates the glycine concentration in mg/cm^3.
    
    Parameters
    ----------
    n_gly : int
        Number of glycine molecules.
    box : list
        Side lengths of the simulation box [a, b, c] in nm.

    Returns
    -------
    conc_gly : float
        Glycine concentration in mg/cm^3.
    """
    box_vol = np.prod(box * ureg.nm)
    conc_gly = (n_gly / box_vol) * (GLY_MOLAR_MASS / ureg("N_A"))
    return conc_gly.to("mg/cm^3").magnitude