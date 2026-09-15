import numpy as np
import pandas as pd
import pytest

from simutils.analysis.blocking import blocking_analysis, optimal_block


def test_blocking_analysis_constant_series_has_zero_sem():
    series = pd.Series(np.full(16, 5.0))

    blocking_df = blocking_analysis(series)

    assert (blocking_df["mean"] == 5.0).all()
    assert (blocking_df["sem"] == 0.0).all()


def test_optimal_block_picks_a_valid_level_and_reports_true_mean():
    series = pd.Series(np.random.default_rng(42).normal(loc=10.0, scale=1.0, size=256))
    blocking_df = blocking_analysis(series)

    mean, sem = optimal_block(blocking_df)

    assert mean in blocking_df["mean"].values
    assert sem in blocking_df["sem"].values
    assert mean == pytest.approx(10.0, abs=0.5)
