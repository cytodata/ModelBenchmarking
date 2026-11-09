"""Load, filter and summarize morphology profiles before pairing them with DRUG-seq.

These helpers implement the per-model steps of the Step 4 benchmark: cache a model's
table from Cloud Storage, read only the treated wells of the compounds of interest,
find grouping and feature columns, and take the median of replicate wells.
"""

import os
import subprocess

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Candidate column names for dose, time and cell line, in order of preference.
GROUP_COLUMN_CANDIDATES = [
    ["concentration", "Metadata_Concentration", "pert_dose", "pert_dose_uM", "dose"],
    ["Metadata_Time", "pert_time", "pert_time_h", "time", "timepoint"],
    ["Metadata_cell_line", "Metadata_CellLine", "Metadata_cell_type", "cell_line", "cell_type"],
]


def cache_filename(name: str) -> str:
    """Local cache file name for a model, e.g. ``"DINO (Raw)"`` -> ``"cached_DINO_Raw.parquet"``."""
    clean_tag = (
        name.replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "").replace("-", "_").replace(",", "")
    )
    return f"cached_{clean_tag}.parquet"


def cache_from_gcs(uri: str, cache_name: str, name: str) -> bool:
    """Download ``uri`` to ``cache_name`` with ``gcloud storage cp`` unless it is already cached.

    Empty (0-byte) cache files left by interrupted downloads are removed first.

    Parameters
    ----------
    uri : str
        ``gs://`` source.
    cache_name : str
        Local destination.
    name : str
        Model name used in messages.

    Returns
    -------
    bool
        ``True`` if the file is available locally, ``False`` if the download failed.
    """
    if os.path.exists(cache_name) and os.path.getsize(cache_name) == 0:
        os.remove(cache_name)

    if not os.path.exists(cache_name):
        print(f"\n📥 Downloading {name} from GCS ({uri})...")
        cmd = ["gcloud", "storage", "cp", uri, cache_name]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  ❌ Failed to download {name}: {res.stderr}")
            if os.path.exists(cache_name):
                os.remove(cache_name)
            return False
        print(f"  ✅ Cached {cache_name} ({os.path.getsize(cache_name) / 1e6:.1f} MB)")
    return True


def read_treated_wells(
    conn: duckdb.DuckDBPyConnection, cache_name: str, name: str, key_view: str, merge_key: str = "InChIKey"
) -> pd.DataFrame | None:
    """Read the treated wells of the compounds listed in a DuckDB view.

    Only rows whose InChIKey appears in ``key_view`` (a registered DuckDB view with a
    ``merge_key`` column) are read from disk; if the table has ``Metadata_pert_type``,
    only treated wells (``'trt'``) are kept. The InChIKey column is renamed to
    ``merge_key``.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
    cache_name : str
        Local Parquet file.
    name : str
        Model name used in messages.
    key_view : str
        Name of the DuckDB view holding the compounds to keep.
    merge_key : str, default "InChIKey"

    Returns
    -------
    pandas.DataFrame or None
        ``None`` (after printing why) if the file cannot be read, has no InChIKey column,
        the query fails, or no compound matches.
    """
    try:
        schema_fields = [f.name for f in pq.read_schema(cache_name)]
    except Exception as e:
        print(f"  ❌ [CORRUPT FILE] Could not read schema for {name}: {e}. Skipping.")
        return None

    ikey_col = next((c for c in schema_fields if "inchikey" in c.lower()), None)
    if not ikey_col:
        print(f"  ❌ No InChIKey found in {name}. Skipping.")
        return None

    has_pert_type = "Metadata_pert_type" in schema_fields
    pert_filter = "AND m.Metadata_pert_type = 'trt'" if has_pert_type else ""

    # Single-threaded read limits in-flight row-group decompression (memory)
    conn.execute("SET threads = 1;")
    conn.execute("SET preserve_insertion_order = false;")

    pushdown_query = f"""
        SELECT *
        FROM read_parquet('{cache_name}') m
        WHERE {ikey_col} IN (SELECT DISTINCT {merge_key} FROM {key_view})
        {pert_filter}
    """
    try:
        morph_df = conn.execute(pushdown_query).df()
    except Exception as e:
        print(f"  ❌ Failed to query {name}: {e}. Skipping.")
        return None
    finally:
        conn.execute("SET threads = 4;")

    if ikey_col != merge_key:
        morph_df = morph_df.rename(columns={ikey_col: merge_key})

    if morph_df.empty:
        print(f"  ❌ Zero overlapping compounds with DRUG-seq for {name}. Skipping.")
        return None
    return morph_df


def detect_group_columns(df: pd.DataFrame, merge_key: str = "InChIKey") -> list[str]:
    """Columns that define one treatment: the compound plus its dose, time and cell line if present.

    For each of dose, time and cell line, the first matching name from
    :data:`GROUP_COLUMN_CANDIDATES` is used.
    """
    group_cols = [merge_key]
    for candidates in GROUP_COLUMN_CANDIDATES:
        for c in candidates:
            if c in df.columns and c not in group_cols:
                group_cols.append(c)
                break
    return group_cols


def morphology_feature_columns(df: pd.DataFrame, merge_key: str, group_cols: list[str]) -> list[str]:
    """Numeric feature columns: excludes ``Metadata_*``, identifiers and grouping columns."""
    return [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if not c.startswith("Metadata_") and c not in [merge_key, "pubchem_cid"] + group_cols
    ]


def median_by_group_duckdb(
    conn: duckdb.DuckDBPyConnection, df: pd.DataFrame, group_cols: list[str], feature_cols: list[str]
) -> pd.DataFrame:
    """Median of every feature per treatment group, computed by DuckDB (sorted by group)."""
    conn.execute("SET threads = 4;")
    median_exprs = ", ".join([f'MEDIAN("{c}") AS "{c}"' for c in feature_cols])
    group_by_sql = ", ".join([f'"{c}"' for c in group_cols])

    conn.register("raw_morph_wells", df)
    result = conn.execute(f"""
        SELECT
            {group_by_sql},
            {median_exprs}
        FROM raw_morph_wells
        GROUP BY {group_by_sql}
        ORDER BY {group_by_sql}
    """).df()
    conn.unregister("raw_morph_wells")
    return result


def adjusted_svcca_auc(true_corrs: np.ndarray, null_mean: np.ndarray, max_axes: int) -> tuple[float, float, float]:
    """Mean canonical correlation over the first ``max_axes`` axes, raw and null-adjusted.

    Returns
    -------
    true_auc, null_auc, adjusted_auc : float
        ``adjusted_auc = true_auc - null_auc``.
    """
    eval_len = min(max_axes, len(true_corrs))
    true_auc = float(np.mean(true_corrs[:eval_len]))
    null_auc = float(np.mean(null_mean[:eval_len]))
    return true_auc, null_auc, true_auc - null_auc
