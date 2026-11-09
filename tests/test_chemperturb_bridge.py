"""Tests for model_benchmarking.chemperturb_bridge (featurization and encoder, no cloud access)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from model_benchmarking import chemperturb_bridge as cpb

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CPU = torch.device("cpu")


def test_split_gcs_uri() -> None:
    """A gs:// URI is split into its bucket and object path."""
    assert cpb.split_gcs_uri("gs://bucket/a/b/c.parquet") == ("bucket", "a/b/c.parquet")


def test_featurize_smiles_valid_molecule() -> None:
    """A valid SMILES yields its canonical SMILES, InChIKey and a binary 2,048-bit fingerprint."""
    can_smi, ikey, fp = cpb.featurize_smiles(ASPIRIN)
    assert can_smi == ASPIRIN
    assert ikey == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    assert fp.shape == (2048,) and fp.dtype == np.float32
    assert set(np.unique(fp)) <= {0.0, 1.0} and fp.sum() > 0


@pytest.mark.parametrize("smiles", [None, "", "   ", "not-a-smiles"])
def test_featurize_smiles_invalid_input(smiles: str | None) -> None:
    """Missing or unparsable SMILES return three Nones instead of raising."""
    assert cpb.featurize_smiles(smiles) == (None, None, None)


def test_find_smiles_column() -> None:
    """The first recognized SMILES column is returned, and a table without one raises."""
    assert cpb.find_smiles_column(pd.DataFrame(columns=["id", "SMILES_standard"])) == "SMILES_standard"
    with pytest.raises(ValueError):
        cpb.find_smiles_column(pd.DataFrame(columns=["id"]))


def test_featurize_compounds_adds_columns() -> None:
    """Featurizing a table adds canonical SMILES and InChIKey columns and one fingerprint per row."""
    df = pd.DataFrame({"SMILES": [ASPIRIN, "invalid"]})
    fps = cpb.featurize_compounds(df, "SMILES")
    assert len(fps) == 2 and fps[1] is None
    assert df["inchikey"].tolist() == ["BSYNRYMUTXBXSQ-UHFFFAOYSA-N", None]


def test_latent_embeddings_align_with_input_rows() -> None:
    """Embeddings have one row per input, with NaN rows where no fingerprint exists."""
    torch.manual_seed(0)
    model = cpb.get_encoder_model(None, latent_dim=8, device=CPU)
    fp = cpb.featurize_smiles(ASPIRIN)[2]
    latent = cpb.compute_latent_embeddings([fp, None, fp], model, device=CPU, latent_dim=8)
    assert latent.shape == (3, 8)
    assert np.isnan(latent[1]).all() and np.isfinite(latent[[0, 2]]).all()
    np.testing.assert_allclose(latent[0], latent[2])


def test_encoder_loads_checkpoint_with_prefixes(tmp_path: Path) -> None:
    """Checkpoint weights saved under a ``compound_encoder.`` prefix are loaded into the encoder."""
    torch.manual_seed(1)
    source = cpb.LPMCompoundEncoder(latent_dim=8)
    state = {f"compound_encoder.{k}": v for k, v in source.state_dict().items()}
    path = tmp_path / "weights.ckpt"
    torch.save({"state_dict": state}, path)

    loaded = cpb.get_encoder_model(str(path), latent_dim=8, device=CPU)
    for name, tensor in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor)
    assert not loaded.training
