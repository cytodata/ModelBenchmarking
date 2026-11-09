"""Collapse several profiles of the same compound into a single profile."""

import numpy as np
import pandas as pd


def max_abs_pool(x: pd.DataFrame) -> np.ndarray:
    """For each feature, keep the value with the largest magnitude across rows.

    Used to merge the DRUG-seq signatures of one compound measured at several doses or
    time points: for every gene, the strongest response (up or down, with its original
    sign) is kept.

    Parameters
    ----------
    x : pandas.DataFrame
        Rows are profiles of one compound, columns are features.

    Returns
    -------
    numpy.ndarray
        One value per column.

    Notes
    -----
    Each feature is chosen independently, so the pooled profile can mix values from
    different doses. Taking the per-feature extreme also favours noise: with more rows
    per compound, the expected maximum absolute value grows even when the compound has
    no effect.
    """
    idx = np.argmax(np.abs(x.values), axis=0)
    return x.values[idx, np.arange(x.shape[1])]


def max_absolute_pooling(df: pd.DataFrame, group_col: str, feature_cols: list[str]) -> pd.DataFrame:
    """Apply :func:`max_abs_pool` to every group (e.g. compound) of a table.

    Parameters
    ----------
    df : pandas.DataFrame
        Long table with one row per profile.
    group_col : str
        Column identifying the group (e.g. ``'InChIKey'``).
    feature_cols : list of str
        Feature columns to pool.

    Returns
    -------
    pandas.DataFrame
        One row per group: the group column followed by the pooled features.
    """
    grouped = df.groupby(group_col)[feature_cols]
    pooled_data = grouped.apply(max_abs_pool)
    return pd.DataFrame(pooled_data.tolist(), index=pooled_data.index, columns=feature_cols).reset_index()
