"""Latent perturbation embeddings of compounds with a ChemPerturb-Bridge-style encoder.

Pipeline: SMILES -> RDKit molecule -> 2,048-bit Morgan fingerprint -> compound encoder
(2,048 -> 512 -> 256) -> embedding. Used by
``notebooks/0.download_and_assemble/0.5.chemperturb_bridge_embeddings.ipynb`` and runnable
from the command line::

    python -m model_benchmarking.chemperturb_bridge --checkpoint /path/to/weights.ckpt

Warning
-------
Without a checkpoint the encoder weights are **random**, so the embeddings are a random
projection of the fingerprints rather than a learned perturbation space. See
``docs/code_review.md`` (C1).
"""

import argparse
import os
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from google.cloud import storage
from rdkit import Chem
from rdkit.Chem import AllChem, inchi
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------
INPUT_GCS_URI = (
    "gs://calico-reference-embeddings/Annotations_from_Roman/"
    "Bayer_data_harmonized/annotations_from_Roman/"
    "JUMP_CP_compound_metadata_with_SMILES_standard.csv"
)

OUTPUT_GCS_URI = (
    "gs://calico-reference-embeddings/embeddings/"
    "ChemPerturb-Bridge/latent_perturbation_embeddings/"
    "jump_cp_latent_perturbation_embeddings.parquet"
)

LOCAL_TMP_DIR = "/content/tmp_perturbridge"
SMILES_COLUMN_CANDIDATES = ["SMILES", "smiles", "SMILES_standard", "canonical_smiles"]
LATENT_DIM = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Google Cloud Storage utilities
# ---------------------------------------------------------------------------
def split_gcs_uri(uri: str) -> tuple[str, str]:
    """Split ``gs://bucket/path/to/blob`` into ``('bucket', 'path/to/blob')``."""
    parsed = urlparse(uri)
    bucket = parsed.netloc
    blob = parsed.path.lstrip("/")
    return bucket, blob


def download_gcs_blob(gcs_uri: str, local_destination: str) -> str:
    """Download one Cloud Storage object to a local file and return the local path."""
    bucket_name, blob_name = split_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    print(f"📥 Downloading {gcs_uri} -> {local_destination}...")
    blob.download_to_filename(local_destination)
    print(f"   Size: {os.path.getsize(local_destination) / (1024 * 1024):.2f} MB")
    return local_destination


def upload_gcs_blob(local_source: str, gcs_uri: str) -> None:
    """Upload a local file to a Cloud Storage object."""
    bucket_name, blob_name = split_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    print(f"📤 Uploading {local_source} -> {gcs_uri}...")
    blob.upload_from_filename(local_source, timeout=600)
    print(f"   Upload complete! Final GCS Size: {blob.size / (1024 * 1024):.2f} MB")


# ---------------------------------------------------------------------------
# Compound encoder
# ---------------------------------------------------------------------------
class LPMCompoundEncoder(nn.Module):
    """Latent Perturbation Model (LPM) compound encoder.

    Maps 2,048-bit Morgan fingerprints into the phenotypic perturbation latent space
    (Z_compound): Linear -> BatchNorm -> GELU -> Dropout -> Linear -> LayerNorm.

    Parameters
    ----------
    input_dim : int, default 2048
    hidden_dim : int, default 512
    latent_dim : int, default 256
    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, latent_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of fingerprints of shape ``(batch, input_dim)``."""
        return self.net(x)


def get_encoder_model(
    checkpoint_path: str | None = None, latent_dim: int = LATENT_DIM, device: torch.device = DEVICE
) -> LPMCompoundEncoder:
    """Build the encoder in evaluation mode, loading weights from ``checkpoint_path`` if it exists.

    Parameters
    ----------
    checkpoint_path : str, optional
        PyTorch / PyTorch Lightning checkpoint. Prefixes ``compound_encoder.`` and
        ``encoder.`` are stripped from parameter names.
    latent_dim : int, default 256
    device : torch.device

    Returns
    -------
    LPMCompoundEncoder

    Notes
    -----
    Weights are loaded with ``strict=False``, which silently skips parameters whose names
    do not match. Without a checkpoint the weights are randomly initialized.
    """
    encoder = LPMCompoundEncoder(input_dim=2048, latent_dim=latent_dim).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"🔑 Loading weights from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        # Strip prefixes if saved from PyTorch Lightning / multi-task model
        cleaned_dict = {}
        for k, v in state_dict.items():
            k_clean = k.replace("compound_encoder.", "").replace("encoder.", "")
            if "net." in k_clean:
                cleaned_dict[k_clean] = v
        encoder.load_state_dict(cleaned_dict, strict=False)
    else:
        print("ℹ️ Checkpoint path not provided or not found. Initializing base LPM compound encoder.")

    encoder.eval()
    return encoder


# ---------------------------------------------------------------------------
# Chemistry featurization
# ---------------------------------------------------------------------------
def featurize_smiles(smiles: str, n_bits: int = 2048) -> tuple[str | None, str | None, np.ndarray | None]:
    """Canonical SMILES, InChIKey and Morgan fingerprint (radius 2) of one molecule.

    Returns ``(None, None, None)`` if the SMILES is missing or cannot be parsed.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None, None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, None
        can_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        ik = inchi.MolToInchiKey(mol)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        return can_smiles, ik, arr
    except Exception:
        return None, None, None


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------
def load_compound_metadata(input_uri: str = INPUT_GCS_URI, tmp_dir: str = LOCAL_TMP_DIR) -> pd.DataFrame:
    """Cache the JUMP compound metadata CSV in ``tmp_dir`` and load it."""
    os.makedirs(tmp_dir, exist_ok=True)
    local_input = os.path.join(tmp_dir, "jump_input.csv")
    if not os.path.exists(local_input) or os.path.getsize(local_input) == 0:
        download_gcs_blob(input_uri, local_input)
    else:
        print(f"📦 Reusing cached raw metadata: {local_input}")
    df = pd.read_csv(local_input)
    print(f"📊 Loaded {len(df):,} JUMP-CP records.")
    return df


def find_smiles_column(df: pd.DataFrame) -> str:
    """Return the first column of ``df`` named like a SMILES column."""
    smiles_col = next((c for c in SMILES_COLUMN_CANDIDATES if c in df.columns), None)
    if not smiles_col:
        raise ValueError(f"Could not locate SMILES column. Columns present: {df.columns.tolist()}")
    return smiles_col


def featurize_compounds(df: pd.DataFrame, smiles_col: str) -> list[np.ndarray | None]:
    """Featurize every SMILES in ``df[smiles_col]``.

    Adds ``canonical_smiles`` and ``inchikey`` columns to ``df`` (in place) and returns
    one fingerprint per row (``None`` where the SMILES could not be parsed).
    """
    print(f"🧪 Featurizing structures from '{smiles_col}' using RDKit...")
    canonical_list, inchikey_list, fp_list = [], [], []

    for s in tqdm(df[smiles_col], desc="Morgan Fingerprinting"):
        can_smi, ik, fp_vec = featurize_smiles(s)
        canonical_list.append(can_smi)
        inchikey_list.append(ik)
        fp_list.append(fp_vec)

    df["canonical_smiles"] = canonical_list
    df["inchikey"] = inchikey_list

    valid_count = sum(fp is not None for fp in fp_list)
    print(f"   Successfully featurized {valid_count:,} / {len(df):,} ({100 * valid_count / len(df):.2f}%) molecules.")
    return fp_list


def compute_latent_embeddings(
    fp_list: list[np.ndarray | None],
    model: LPMCompoundEncoder,
    device: torch.device = DEVICE,
    latent_dim: int = LATENT_DIM,
    batch_size: int = 512,
) -> np.ndarray:
    """Encode the fingerprints in batches; rows without a fingerprint are filled with NaN.

    Returns
    -------
    numpy.ndarray, shape (len(fp_list), latent_dim)
    """
    valid_mask = [fp is not None for fp in fp_list]
    valid_fps = np.stack([fp for fp in fp_list if fp is not None])
    all_latents = []

    with torch.no_grad():
        for i in range(0, len(valid_fps), batch_size):
            batch_np = valid_fps[i : i + batch_size]
            batch_t = torch.from_numpy(batch_np).to(device)
            z = model(batch_t).cpu().numpy()
            all_latents.append(z)

    z_matrix = np.concatenate(all_latents, axis=0)

    # Re-align with full DataFrame (assigning NaNs to failed structures)
    full_latent_array = np.full((len(fp_list), latent_dim), np.nan, dtype=np.float32)
    full_latent_array[np.array(valid_mask)] = z_matrix
    return full_latent_array


def save_and_upload(
    df: pd.DataFrame,
    latent: np.ndarray,
    output_uri: str = OUTPUT_GCS_URI,
    tmp_dir: str = LOCAL_TMP_DIR,
) -> pd.DataFrame:
    """Append the embedding columns to ``df``, write Parquet locally and upload it.

    Returns the combined table (metadata + ``latent_pert_dim_0 ...`` columns).
    """
    latent_cols = [f"latent_pert_dim_{i}" for i in range(latent.shape[1])]
    latent_df = pd.DataFrame(latent, columns=latent_cols, index=df.index)
    result_df = pd.concat([df, latent_df], axis=1)

    os.makedirs(tmp_dir, exist_ok=True)
    local_output = os.path.join(tmp_dir, "jump_cp_latent_perturbation_embeddings.parquet")
    print(f"💾 Saving Parquet table locally: {local_output}...")
    result_df.to_parquet(local_output, engine="pyarrow", compression="snappy", index=False)

    upload_gcs_blob(local_output, output_uri)
    return result_df


def main(checkpoint_path: str | None = None) -> None:
    """Run the full extraction pipeline and upload the result to ``OUTPUT_GCS_URI``."""
    print("=" * 70)
    print("🚀 STARTING JUMP-CP LATENT PERTURBATION EXTRACTION PIPELINE")
    print("=" * 70)

    df = load_compound_metadata()
    smiles_col = find_smiles_column(df)
    fp_list = featurize_compounds(df, smiles_col)

    print(f"🧠 Computing latent embeddings on device: {DEVICE}...")
    model = get_encoder_model(checkpoint_path, latent_dim=LATENT_DIM, device=DEVICE)
    latent = compute_latent_embeddings(fp_list, model)

    save_and_upload(df, latent)

    valid_count = int(np.sum(~np.isnan(latent[:, 0])))
    print("=" * 70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"   Destination URI: {OUTPUT_GCS_URI}")
    print(f"   Total rows with embeddings: {valid_count:,}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to LPM .pt / .ckpt weights")
    args = parser.parse_args()
    main(checkpoint_path=args.checkpoint)
