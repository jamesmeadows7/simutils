import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_timeseries(series: pd.Series, xlabel: str, ylabel: str, mean: float = None, sem: float = None, ax: Axes = None) -> tuple[Figure, Axes]:
    """
    Plots a timeseries showing the mean and standard error of the data if specified.

    Parameters
    ----------
    series : Series
        Timeseries values.
    xlabel : str
        Label text for the horizontal axis.
    ylabel : str
        Label text for the vertical axis.
    mean : float, optional
        Mean value of the timeseries.
    sem : float, optional
        Standard error in the mean value of the timeseries.
    ax : Axis, optional
        Existing axis to plot to.
    
    Returns
    -------
    fig : Figure
        Plot figure.
    ax : Axis
        Plot axis.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(series.index, series)
    if mean is not None:
        ax.axhline(mean, color="tab:red", zorder=3)
    if sem is not None:
        x0, x1 = ax.get_xlim()
        ax.fill_between([x0, x1], mean-sem, mean+sem, color="tab:red", alpha=0.3, zorder=3, lw=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(x0, x1)
    return fig, ax


def plot_blocking_analysis(blocking_df: pd.DataFrame, units: str, sem: float = None, ax: Axes = None) -> tuple[Figure, Axes]:
    """
    Plots a timeseries showing the mean and standard error of the data if specified.

    Parameters
    ----------
    blocking_df : DataFrame
        Number of blocks, mean, sem and sem error for each blocking operation.
    units : str
        Units of the analysed quantity.
    sem : float, optional
        Standard error in the mean value of the timeseries.
    ax : Axis, optional
        Existing axis to plot to.
    
    Returns
    -------
    fig : Figure
        Plot figure.
    ax : Axis
        Plot axis.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.errorbar(blocking_df.index, blocking_df["sem"], yerr=blocking_df["sem_err"])
    if sem is not None:
        ax.axhline(sem, color="tab:red", zorder=3)
    ax.set_xlabel("Blocking Operations")
    ax.set_ylabel(f"σ_m / {units}")
    return fig, ax