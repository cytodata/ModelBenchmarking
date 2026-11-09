"""Load morphology tables and a chosen transcriptomic "anchor" for the Step 4.3 benchmark.

In Step 4.3 the transcriptomic side of every comparison is interchangeable (raw DRUG-seq genes,
DRUG-seq factors or latent perturbation embeddings), and the morphology files do not all name
their InChIKey column the same way. These helpers find the identifier column, validate cached
files, work out which compounds two (or all) datasets share, and stream only those compounds
from disk with DuckDB.
"""

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Collection, Mapping

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from model_benchmarking.pooling import max_absolute_pooling

DMSO_INCHIKEY = "IAZDPXIOMUYVGZ-UHFFFAOYSA-N"
RNA_METADATA_COLUMNS = ["pubchem_cid", "pert_time_h", "pert_dose_uM"]
MORPHOLOGY_EXCLUDED_COLUMNS = ["pubchem_cid"]


def detect_inchikey_col(file_path: str) -> str | None:
    """Name of the InChIKey column of a Parquet file, or ``None`` if there is none.

    Column names are searched first (ignoring truncated ``*14`` keys and ``prefix`` columns),
    then the pandas index stored in the Parquet metadata, then ``Metadata_InChIKey``.
    """
    schema = pq.read_schema(file_path)
    col_names = [f.name for f in schema]

    key = next(
        (
            c
            for c in col_names
            if "inchikey" in c.lower() and not c.lower().endswith("14") and "prefix" not in c.lower()
        ),
        None,
    )
    if key:
        return key

    if schema.metadata and b"pandas" in schema.metadata:
        try:
            pandas_meta = json.loads(schema.metadata[b"pandas"].decode("utf-8"))
            for idx in pandas_meta.get("index_columns", []):
                if isinstance(idx, str) and ("inchikey" in idx.lower() or idx in col_names):
                    return idx
        except (ValueError, UnicodeDecodeError):
            pass

    return "Metadata_InChIKey" if "Metadata_InChIKey" in col_names else None


def resolve_key_column(columns: Collection[str]) -> str | None:
    """Pick the InChIKey column among ``columns`` (exact known names first, then any ``*inchikey*``)."""
    for candidate in ("InChIKey", "inchikey", "Metadata_InChIKey"):
        if candidate in columns:
            return candidate
    return next((c for c in columns if "inchikey" in c.lower()), None)


def standardize_inchikey_column(df: pd.DataFrame, target_col: str = "InChIKey") -> pd.DataFrame:
    """Rename the InChIKey column to ``target_col`` and drop any other InChIKey-like columns.

    Raises
    ------
    ValueError
        If ``df`` has no InChIKey column.
    """
    key_col = resolve_key_column(list(df.columns))
    if not key_col:
        raise ValueError(f"No InChIKey found in DataFrame! Available: {list(df.columns)[:10]}")

    redundant = [c for c in df.columns if "inchikey" in c.lower() and c not in (key_col, target_col)]
    if redundant:
        df = df.drop(columns=redundant)

    if key_col != target_col:
        df = df.drop(columns=[key_col]) if target_col in df.columns else df.rename(columns={key_col: target_col})
    return df


def cache_name_for_uri(uri: str) -> str:
    """Local cache file name derived from a ``gs://`` URI, so that two sources never share a cache.

    The name contains the parent folder and the file name, e.g.
    ``gs://bucket/indexed_by_inchikey/a.parquet`` -> ``cached_indexed_by_inchikey__a.parquet``. Changing
    the URI of a dataset therefore never reuses another dataset's download.
    """
    parts = uri.rstrip("/").split("/")
    return f"cached_{parts[-2]}__{parts[-1]}" if len(parts) >= 2 else f"cached_{parts[-1]}"


def cache_validated_from_gcs(uri: str, cache_name: str, name: str) -> None:
    """Make sure ``cache_name`` holds a usable copy of ``uri``, downloading it if needed.

    A cached file that is empty or has no InChIKey column (a stale or interrupted download) is
    deleted and fetched again with ``gcloud storage cp``.

    Raises
    ------
    subprocess.CalledProcessError
        If the download fails.
    """
    if os.path.exists(cache_name) and (os.path.getsize(cache_name) == 0 or detect_inchikey_col(cache_name) is None):
        print(f"  🔄 Invalidating stale cache for [{name}] (empty or missing InChIKey)...")
        os.remove(cache_name)

    if not os.path.exists(cache_name):
        print(f"  📥 Pre-caching [{name}] from GCS ({uri})...")
        subprocess.run(["gcloud", "storage", "cp", uri, cache_name], check=True)


def _treated_clause(schema_cols: list[str], morph_pert_filter: bool = True) -> str:
    """Return the SQL fragment that keeps treated wells (``trt``), if the table says which wells are treated."""
    return "AND Metadata_pert_type = 'trt'" if ("Metadata_pert_type" in schema_cols and morph_pert_filter) else ""


def _rna_time_clause(rna_schema: list[str]) -> str:
    """Return the SQL fragment that keeps the 24 h and 48 h time points, if the anchor has a time column."""
    time_col = next((c for c in rna_schema if "pert_time" in c.lower()), None)
    return f"AND TRY_CAST({time_col} AS FLOAT) IN (24, 48)" if time_col else ""


def find_common_compounds(
    morphology_paths: Mapping[str, str], rna_path: str, dmso_inchikey: str = DMSO_INCHIKEY
) -> set[str]:
    """InChIKeys present in the anchor and in *every* morphology file (DMSO excluded).

    Parameters
    ----------
    morphology_paths : mapping of str to str
        Model name -> local Parquet file.
    rna_path : str
        Local Parquet file of the transcriptomic anchor.
    dmso_inchikey : str
        Solvent control to exclude.

    Returns
    -------
    set of str
        Compounds every model can be evaluated on (the "common cohort").
    """
    conn = duckdb.connect()
    try:
        rna_key = detect_inchikey_col(rna_path)
        time_clause = _rna_time_clause([f.name for f in pq.read_schema(rna_path)])
        common = set(
            conn.execute(
                f"""
                SELECT DISTINCT {rna_key} FROM read_parquet('{rna_path}')
                WHERE {rna_key} IS NOT NULL AND {rna_key} != '{dmso_inchikey}' {time_clause}
                """
            ).df()[rna_key]
        )

        for name, path in morphology_paths.items():
            morph_key = detect_inchikey_col(path)
            schema = [f.name for f in pq.read_schema(path)]
            pert_filter = (
                "WHERE Metadata_pert_type = 'trt'"
                if "Metadata_pert_type" in schema
                else f"WHERE {morph_key} != '{dmso_inchikey}'"
            )
            keys = set(
                conn.execute(
                    f"SELECT DISTINCT {morph_key} FROM read_parquet('{path}') {pert_filter} AND {morph_key} IS NOT NULL"
                ).df()[morph_key]
            )
            common &= keys
            print(f"  ↳ Intersected [{name:<35}] ➔ Retained: {len(common):,} compounds")
    finally:
        conn.close()
    return common


def load_overlap_pair(
    morph_path: str,
    rna_path: str,
    target_keys: Collection[str] | None = None,
    dmso_inchikey: str = DMSO_INCHIKEY,
    merge_key: str = "InChIKey",
    morph_pert_filter: bool = True,
    temp_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream the compounds shared by a morphology file and a transcriptomic anchor.

    Only matching compounds are decompressed (DuckDB predicate pushdown). Treated wells of
    each compound are collapsed with the **median**, anchor rows of one compound (several doses
    or time points) with **max-absolute pooling**, and both matrices are returned in the same,
    sorted, compound order. If the anchor has a time column, only 24 h and 48 h rows are used.

    Parameters
    ----------
    morph_path, rna_path : str
        Local Parquet files.
    target_keys : collection of str, optional
        Restrict the comparison to these compounds (e.g. a common cohort). By default, all
        compounds present in both files, excluding DMSO.
    dmso_inchikey : str
        Solvent control to exclude.
    merge_key : str, default "InChIKey"
        Name used for the compound identifier.
    morph_pert_filter : bool, default True
        Keep only treated wells when the morphology table has ``Metadata_pert_type``.
    temp_dir : str, optional
        DuckDB spill folder; a fresh temporary folder by default.

    Returns
    -------
    X_morph : numpy.ndarray, float32, shape (n_compounds, n_morphology_features)
    Y_rna : numpy.ndarray, float32, shape (n_compounds, n_anchor_features)
    inchikeys : numpy.ndarray of str
        Compound of each row.

    Raises
    ------
    ValueError
        If a file has no InChIKey column or the two files share no compound.
    """
    owns_temp = temp_dir is None
    temp_dir = temp_dir or tempfile.mkdtemp(prefix="duckdb_load_")
    conn = duckdb.connect()
    conn.execute(f"PRAGMA temp_directory='{temp_dir}';")
    conn.execute("SET preserve_insertion_order = false;")

    try:
        morph_schema = [f.name for f in pq.read_schema(morph_path)]
        rna_schema = [f.name for f in pq.read_schema(rna_path)]
        morph_key = detect_inchikey_col(morph_path)
        rna_key = detect_inchikey_col(rna_path)
        if not morph_key or not rna_key:
            raise ValueError(
                f"InChIKey column missing! Morph ({os.path.basename(morph_path)}): {morph_key} "
                f"(columns {morph_schema[:8]}...) | RNA ({os.path.basename(rna_path)}): {rna_key} "
                f"(columns {rna_schema[:8]}...)"
            )

        pert_clause = _treated_clause(morph_schema, morph_pert_filter)
        time_clause = _rna_time_clause(rna_schema)

        if target_keys is not None:
            shared_keys = pd.DataFrame({"ikey": sorted(target_keys)})
        else:
            shared_keys = conn.execute(
                f"""
                WITH morph_keys AS (
                    SELECT DISTINCT {morph_key} AS ikey FROM read_parquet('{morph_path}')
                    WHERE {morph_key} IS NOT NULL AND {morph_key} != '{dmso_inchikey}' {pert_clause}
                ),
                rna_keys AS (
                    SELECT DISTINCT {rna_key} AS ikey FROM read_parquet('{rna_path}')
                    WHERE {rna_key} IS NOT NULL AND {rna_key} != '{dmso_inchikey}' {time_clause}
                )
                SELECT ikey FROM morph_keys INTERSECT SELECT ikey FROM rna_keys ORDER BY ikey ASC
                """
            ).df()

        if len(shared_keys) == 0:
            raise ValueError("Zero overlapping compounds found between datasets!")
        conn.register("shared_keys_view", shared_keys)

        morph_df = conn.execute(
            f"""
            SELECT * FROM read_parquet('{morph_path}')
            WHERE {morph_key} IN (SELECT ikey FROM shared_keys_view) {pert_clause}
            """
        ).df()
        rna_df = conn.execute(
            f"""
            SELECT * FROM read_parquet('{rna_path}')
            WHERE {rna_key} IN (SELECT ikey FROM shared_keys_view) {time_clause}
            """
        ).df()
    finally:
        conn.close()
        if owns_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)

    morph_df = standardize_inchikey_column(morph_df, merge_key)
    rna_df = standardize_inchikey_column(rna_df, merge_key)

    morph_features = [
        c
        for c in morph_df.select_dtypes(include=[np.number]).columns
        if not c.startswith(("Metadata_", "__")) and c not in [merge_key, *MORPHOLOGY_EXCLUDED_COLUMNS]
    ]
    rna_features = [
        c
        for c in rna_df.select_dtypes(include=[np.number]).columns
        if not c.startswith(("Metadata_", "__")) and c not in [merge_key, *RNA_METADATA_COLUMNS]
    ]

    morph_consensus = (
        morph_df.groupby(merge_key, as_index=False, dropna=False)[morph_features].median().sort_values(merge_key)
    )
    if rna_df[merge_key].is_unique:
        rna_consensus = rna_df[[merge_key, *rna_features]]
    else:
        sort_cols = [c for c in [merge_key, "pert_dose_uM", "pert_time_h"] if c in rna_df.columns]
        rna_consensus = max_absolute_pooling(rna_df.sort_values(by=sort_cols, kind="stable"), merge_key, rna_features)

    morph_consensus = morph_consensus.sort_values(by=merge_key, kind="stable").reset_index(drop=True)
    rna_consensus = rna_consensus.sort_values(by=merge_key, kind="stable").reset_index(drop=True)

    if not np.array_equal(morph_consensus[merge_key].values, rna_consensus[merge_key].values):
        raise ValueError("Compound ordering mismatch between morphology and the transcriptomic anchor.")

    X_morph = np.nan_to_num(morph_consensus[morph_features].values.astype(np.float32))
    Y_rna = np.nan_to_num(rna_consensus[rna_features].values.astype(np.float32))
    return X_morph, Y_rna, morph_consensus[merge_key].values
