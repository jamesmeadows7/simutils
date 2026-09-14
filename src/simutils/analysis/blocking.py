"""Tools for performing an automated blocking analysis on fluctuating timeseries."""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def blocking_analysis(series: pd.Series) -> pd.DataFrame:
    """
    Performs a blocking procedure on the last 2^N values of a Series, returning a DataFrame of statistics for each blocking operation.

    Parameters
    ----------
    series : Series
        Timeseries values.

    Returns
    -------
    blocking_df : DataFrame
        Number of blocks, mean, sem and sem error for each blocking operation.
    """
    # trim data to last 2^N data points
    max_operations = int(np.log2(len(series)))
    n = 2**max_operations
    series = series.iloc[-n:]
    # initialise results DataFrame
    blocking_df = pd.DataFrame(index=range(max_operations), columns=["N", "mean", "sem", "sem_err"])
    # blocking procedure
    for o in range(max_operations):
        # compute statistics
        N = len(series)
        mean = series.mean()
        sem = series.sem()
        sem_err = series.sem() / np.sqrt(2*(N-1))
        blocking_df.loc[o, "N"] = N
        blocking_df.loc[o, "mean"] = mean
        blocking_df.loc[o, "sem"] = sem
        blocking_df.loc[o, "sem_err"] = sem_err
        # average adjacent values
        series = series.groupby(np.arange(N) // 2).mean()
    return blocking_df


def optimal_block(blocking_df: pd.DataFrame) -> tuple[float, float]:
    """
    Finds the optimal block size in the blocking analysis, Calculates the mean and sem associated with this block size.

    Parameters
    ----------
    blocking_df : DataFrame
        Number of blocks, mean, sem and sem error for each blocking operation.

    Returns
    -------
    mean : int
        Mean value of the timeseries.
    sem : int
        Standard error in the mean value of the timeseries.
    """
    n = blocking_df.loc[0, "N"]
    # compute block size
    block_size = n / blocking_df["N"]
    # compute optimal block size according to 10.1103/PhysRevE.83.066706
    opt_block_size = (2*n*(blocking_df["sem"]/blocking_df["sem"].loc[0])**4)**(1/3)
    # find smallest B that is less than B_opt (this assumes opt_block_size is monotonic in the forward direction)
    opt_idx = (block_size > opt_block_size).idxmax()
    # check optimal B < n / 50
    if block_size[opt_idx] > n / 50:
        logger.info(f"Optimal block size is {block_size[opt_idx]}. Only {n/block_size[opt_idx]} blocks contributing to average. Gather more data for better statistics.")
    mean = blocking_df["mean"].iloc[opt_idx]
    sem = blocking_df["sem"].iloc[opt_idx]
    return mean, sem