"""Two-dimensional chemical similarity of compounds (ECFP4 fingerprints and Tanimoto similarity).

Used to ask whether the compounds driving an axis share a chemical scaffold: if they resemble
each other chemically, the axis may reflect one chemical series; if they do not, different
chemistry converges on the same cellular state.
"""

import os
import subprocess

import duckdb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

# ECFP4: Morgan fingerprint with radius 2, 2048 bits
_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def ensure_smiles_lookup(csv_uri: str, local_parquet: str) -> str:
    """Build (once) a small local Parquet table mapping InChIKey to SMILES from a metadata CSV.

    Parameters
    ----------
    csv_uri : str
        ``gs://`` URI of a CSV with one InChIKey column and one SMILES column (found by name).
    local_parquet : str
        Output path; reused if it already exists.

    Returns
    -------
    str
        ``local_parquet``.

    Raises
    ------
    ValueError
        If the CSV has no InChIKey or SMILES column.
    """
    if os.path.exists(local_parquet):
        return local_parquet

    raw_csv = f"{local_parquet}.tmp.csv"
    print("📥 Pre-caching standardized SMILES metadata...")
    subprocess.run(["gcloud", "storage", "cp", csv_uri, raw_csv], check=True)

    con = duckdb.connect()
    try:
        cols = [c[0] for c in con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{raw_csv}')").fetchall()]
        ikey_col = next((c for c in cols if "inchikey" in c.lower()), None)
        smiles_col = next((c for c in cols if "smiles" in c.lower()), None)
        if not ikey_col or not smiles_col:
            raise ValueError(f"Could not locate InChIKey or SMILES in CSV columns: {cols}")

        con.execute(
            f"""
            COPY (
                SELECT DISTINCT {ikey_col} AS InChIKey, {smiles_col} AS SMILES
                FROM read_csv_auto('{raw_csv}')
                WHERE {ikey_col} IS NOT NULL AND {smiles_col} IS NOT NULL AND {smiles_col} != ''
            ) TO '{local_parquet}' (FORMAT PARQUET, COMPRESSION 'SNAPPY')
            """
        )
    finally:
        con.close()
        if os.path.exists(raw_csv):
            os.remove(raw_csv)
    print("✅ SMILES lookup ready.")
    return local_parquet


def pairwise_tanimoto(smiles: list[str]) -> np.ndarray:
    """ECFP4 Tanimoto similarity of every pair of molecules (upper triangle, ``i < j``).

    SMILES that RDKit cannot parse are skipped.
    """
    fingerprints = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fingerprints.append(_MORGAN_GENERATOR.GetFingerprint(mol))

    n = len(fingerprints)
    return np.array(
        [DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j]) for i in range(n) for j in range(i + 1, n)]
    )


def driver_tanimoto(inchikeys: list[str], smiles_parquet: str) -> tuple[float, float, float, int]:
    """Pairwise ECFP4 Tanimoto statistics of a group of compounds.

    Parameters
    ----------
    inchikeys : list of str
        Compounds to compare.
    smiles_parquet : str
        Table from :func:`ensure_smiles_lookup`.

    Returns
    -------
    mean, median, maximum : float
        Over all pairs (0.0 if fewer than two molecules could be resolved).
    n_valid : int
        Number of compounds with a SMILES that RDKit could parse.
    """
    unique = list(dict.fromkeys(inchikeys))
    con = duckdb.connect()
    try:
        con.register("keys_df", pd.DataFrame({"InChIKey": unique}))
        res = con.execute(
            f"SELECT InChIKey, SMILES FROM read_parquet('{smiles_parquet}') WHERE InChIKey IN (SELECT InChIKey FROM keys_df)"
        ).df()
    finally:
        con.close()

    smiles_map = dict(zip(res["InChIKey"], res["SMILES"], strict=True))
    smiles = [smiles_map[k] for k in unique if k in smiles_map]
    n_valid = sum(Chem.MolFromSmiles(s) is not None for s in smiles)

    sims = pairwise_tanimoto(smiles)
    if len(sims) == 0:
        return 0.0, 0.0, 0.0, n_valid
    return float(np.mean(sims)), float(np.median(sims)), float(np.max(sims)), n_valid


def chemistry_verdict(bioactivity_p_value: float, mean_tanimoto: float) -> str:
    """Interpret driver similarity in bioactivity space together with 2D structural similarity.

    Parameters
    ----------
    bioactivity_p_value : float
        One-sided Mann-Whitney p-value for driver vs. background similarity in the
        bioactivity space (Chemical Checker).
    mean_tanimoto : float
        Mean pairwise ECFP4 Tanimoto similarity of the drivers.
    """
    if bioactivity_p_value < 0.05 and mean_tanimoto >= 0.35:
        return "🔬 CONGENERIC SERIES (Shared 2D chemical scaffold driving common target)"
    if bioactivity_p_value < 0.01 and mean_tanimoto < 0.25:
        return "💡 TRUE PHENOTYPIC CONVERGENCE (Diverse 2D scaffolds hopping to identical bio-state)"
    if bioactivity_p_value < 0.05:
        return "⚠️ WEAK BIOACTIVITY CLUSTERING (Broad target network)"
    return "❓ UNSTRUCTURED MATTER (Chemically and functionally diverse)"
