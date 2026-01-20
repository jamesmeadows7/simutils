import numpy as np
import logging
import panedr
import scipy.signal as signal
from scipy.integrate import cumulative_trapezoid
import tidynamics
from simutils import ureg
from os import PathLike


logger = logging.getLogger(__name__)

def get_box_volume(edr_file: PathLike) -> float:
    """
    Reads final box volume from edr file.

    Parameters
    ----------
    edr_file : PathLike
        Path of the edr file.

    Returns
    -------
    vol : float
        Box volume in nm^3.
    """
    edr = panedr.edr_to_df(edr_file)
    vol = edr["Volume"].iloc[-1]* ureg("nm^3")
    logger.info(f"Volume: {vol.magnitude} nm^3.")
    return vol


def get_pressure_tensor(edr_file: PathLike) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads pressure tensor from edr file.

    Parameters
    ----------
    edr_file : PathLike
        Path of the edr file.

    Returns
    -------
    t : ndarray
        N x 1 array of simulation time in ps.
    p_tensor : ndarray
        N x 3 x 3 pressure tensor in bar.
    """
    logger.info(f"Reading edr file.")
    edr = panedr.edr_to_df(edr_file)
    t = edr["Time"].to_numpy()
    press = edr[["Pres-XX", "Pres-XY", "Pres-XZ", "Pres-YX", "Pres-YY", "Pres-YZ", "Pres-ZX", "Pres-ZY", "Pres-ZZ"]]
    p_tensor = press.to_numpy().reshape(-1, 3, 3)
    return t, p_tensor


def transform_pressure_tensor(p_tensor: np.ndarray) -> np.ndarray:
    """
    Symmetrises pressure tensor and subtracts pressure form diagonal.

    Parameters
    ----------
    p_tensor : ndarray
        N x 3 x 3 pressure tensor in bar.

    Returns
    -------
    p_tensor : ndarray
        N x 3 x 3 transformed pressure tensor in bar.
    """
    logger.info(f"Transforming pressure tensor.")
    p_tensor = (p_tensor + np.transpose(p_tensor, axes=(0, 2, 1))) / 2
    p = np.trace(p_tensor, axis1=1, axis2=2) / 3.0
    p_tensor = p_tensor - (p[:, None, None] * np.eye(3)[None, :, :])
    return p_tensor


def autocorrelation(p_ij: np.ndarray) -> np.ndarray:
    """
    Computes autocorrelation of timeseries using scipy.

    Parameters
    ----------
    p_ij : ndarray
        N x 1 array of pressure values.

    Returns
    -------
    acf : ndarray
        Autocorrelation function of the input time series for positive lagtimes. Normalised by the number of points contributing to each lagtime.
    """
    n = len(p_ij)
    acf = signal.correlate(p_ij, p_ij)[n-1:]
    acf /= np.arange(n, 0, -1)
    return acf


def autocorrelation_fft(p_ij: np.ndarray) -> np.ndarray:
    """
    Computes autocorrelation of timeseries using tidynamics FFT method.

    Parameters
    ----------
    p_ij : ndarray
        N x 1 array of pressure values.

    Returns
    -------
    acf : ndarray
        Autocorrelation function of the input time series for positive lagtimes.
    """
    n = len(p_ij)
    acf = tidynamics.correlation(p_ij, p_ij)[n-1:]
    return acf


def compute_autocorrelation(t: np.ndarray, p_tensor: np.ndarray, max_tau: float, acf_method: str ="fft") -> tuple[np.ndarray, np.ndarray]:
    """
    Computes autocorrelations for each component of the pressure tensor.

    Parameters
    ----------
    t : ndarray
        N x 1 array of simulation time in ps.
    p_tensor : ndarray
        N x 3 x 3 pressure tensor in bar.
    max_tau : float
        Maximum lagtime in ps.
    acf_method : string, optional
        Method to compute autocorrelation: scipy or fft (default = fft).

    Returns
    -------
    tau : ndarray
        N x 1 array of lagtimes.
    acf_tensor : ndarray
        N x 3 x 3 tensor of pressure autocorrelations.
    """
    logger.info(f"Computing autoorrelation for each pressure tensor component.")
    idx = np.argmin((np.abs(t-max_tau)))
    tau = t[:idx]
    acf_tensor = np.zeros((len(tau), 3, 3))
    for i in range(3):
        for j in range(i+1):
            if acf_method == "scipy":
                acf = autocorrelation(p_tensor[:, i, j])
            elif acf_method == "fft":
                acf = autocorrelation_fft(p_tensor[:, i, j])
            acf_tensor[:, i, j] = acf[:idx]
            acf_tensor[:, j, i] = acf[:idx]
    return tau, acf_tensor


def compute_viscosity_tensor(tau: np.ndarray, acf_tensor: np.ndarray, vol: float, temp: float = 298.15) -> np.ndarray:
    """
    Computes by cumulatively integrating the pressure tensor autocorrelations.

    Parameters
    ----------
    tau : ndarray
        N x 1 array of lagtimes.
    acf_tensor : ndarray
        N x 3 x 3 tensor of pressure autocorrelations.
    vol : float
        Box volume in nm^3.
    temp : float, optional
        Simulation temperature in K (default = 298.15).

    Returns
    -------
    eta_tensor : ndarray
        N x 3 x 3 tensor of cumulative viscosity in mPa*s.
    """
    logger.info(f"Computing viscosities.")
    integral = cumulative_trapezoid(acf_tensor, tau, axis=0) * ureg("bar^2*ps")
    eta_tensor = (vol / (ureg("k_B") * temp * ureg("K"))) * integral
    return eta_tensor.to("mPa*s").magnitude