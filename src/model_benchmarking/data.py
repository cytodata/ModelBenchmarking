"""Small data-handling helpers shared across notebooks."""

import pandas as pd


def safely_set_inchikey_index(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Make the InChIKey the row index of an embedding table.

    Every dataset in this project identifies compounds by their InChIKey, but the column
    is spelled differently across sources (``InChIKey``, ``Metadata_InChIKey``, ...) and
    is sometimes already the index. This function finds it and names the index
    ``'InChIKey'`` so tables can be aligned row by row.

    Parameters
    ----------
    df : pandas.DataFrame
        Table that contains an InChIKey column or index.
    name : str
        Dataset name, used only in the error message.

    Returns
    -------
    pandas.DataFrame
        The table indexed by InChIKey (index name ``'InChIKey'``).

    Raises
    ------
    KeyError
        If no InChIKey column or index can be found.

    Notes
    -----
    An unnamed index is assumed to hold InChIKeys when its first value is a string of at
    least 25 characters (standard InChIKeys are 27 characters).
    """
    if df.index.name and "inchikey" in str(df.index.name).lower():
        df.index.name = "InChIKey"
        return df
    if df.index.name is None and isinstance(df.index[0], str) and len(str(df.index[0])) >= 25:
        df.index.name = "InChIKey"
        return df
    inchikey_col = next((c for c in df.columns if "inchikey" in c.lower()), None)
    if inchikey_col:
        df = df.set_index(inchikey_col)
        df.index.name = "InChIKey"
        return df
    raise KeyError(f"❌ Could not find an InChIKey column in {name}! Available columns: {list(df.columns[:15])}")
