"""Tests for model_benchmarking.chemistry (ECFP4 Tanimoto similarity of driver compounds)."""

from pathlib import Path

import pandas as pd
import pytest

from model_benchmarking import chemistry

ETHANOL, METHANOL, PROPANOL, BENZENE = "CCO", "CO", "CCCO", "c1ccccc1"


def test_pairwise_tanimoto_identical_and_dissimilar() -> None:
    """Identical molecules have similarity 1, unrelated molecules far less; bad SMILES are skipped."""
    sims = chemistry.pairwise_tanimoto([ETHANOL, ETHANOL, "not a smiles", BENZENE])
    assert len(sims) == 3  # three valid molecules -> three pairs
    assert sims[0] == pytest.approx(1.0)
    assert sims[1] < 0.2 and sims[2] < 0.2


def test_pairwise_tanimoto_needs_two_molecules() -> None:
    """Fewer than two valid molecules give no pairs."""
    assert len(chemistry.pairwise_tanimoto([ETHANOL])) == 0


@pytest.fixture
def smiles_table(tmp_path: Path) -> str:
    """A small InChIKey -> SMILES lookup table."""
    path = tmp_path / "smiles.parquet"
    pd.DataFrame(
        {"InChIKey": ["K_ETH", "K_METH", "K_PROP", "K_BENZ"], "SMILES": [ETHANOL, METHANOL, PROPANOL, BENZENE]}
    ).to_parquet(path)
    return str(path)


def test_driver_tanimoto(smiles_table: str) -> None:
    """Similar alcohols score higher than a mix with benzene; unresolved compounds are not counted."""
    mean_a, median_a, max_a, n_a = chemistry.driver_tanimoto(["K_ETH", "K_METH", "K_PROP", "K_MISSING"], smiles_table)
    mean_b, _, _, n_b = chemistry.driver_tanimoto(["K_ETH", "K_BENZ"], smiles_table)
    assert (n_a, n_b) == (3, 2)
    assert mean_a > mean_b and max_a >= median_a


def test_driver_tanimoto_without_pairs(smiles_table: str) -> None:
    """A single resolvable compound gives zero similarity and a count of one."""
    assert chemistry.driver_tanimoto(["K_ETH", "K_MISSING"], smiles_table) == (0.0, 0.0, 0.0, 1)


def test_ensure_smiles_lookup_reuses_existing_file(tmp_path: Path) -> None:
    """An existing lookup is returned without downloading anything."""
    existing = tmp_path / "lookup.parquet"
    existing.write_bytes(b"x")
    assert chemistry.ensure_smiles_lookup("gs://bucket/none.csv", str(existing)) == str(existing)


def test_ensure_smiles_lookup_builds_parquet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The CSV is converted to a deduplicated InChIKey/SMILES table and the temporary CSV is removed."""

    def fake_run(cmd: list[str], check: bool) -> None:
        pd.DataFrame(
            {
                "Metadata_InChIKey": ["A", "A", "B", "C", None],
                "smiles_standard": [ETHANOL, ETHANOL, METHANOL, "", PROPANOL],
                "other": range(5),
            }
        ).to_csv(cmd[-1], index=False)

    monkeypatch.setattr(chemistry.subprocess, "run", fake_run)
    out = tmp_path / "lookup.parquet"
    chemistry.ensure_smiles_lookup("gs://bucket/x.csv", str(out))
    table = pd.read_parquet(out).sort_values("InChIKey")
    assert table["InChIKey"].tolist() == ["A", "B"] and table["SMILES"].tolist() == [ETHANOL, METHANOL]
    assert not list(tmp_path.glob("*.csv"))


@pytest.mark.parametrize(
    "p_value, tanimoto, expected",
    [
        (0.001, 0.5, "CONGENERIC"),
        (0.001, 0.1, "TRUE PHENOTYPIC CONVERGENCE"),
        (0.03, 0.1, "WEAK BIOACTIVITY"),
        (0.5, 0.5, "UNSTRUCTURED"),
    ],
)
def test_chemistry_verdict(p_value: float, tanimoto: float, expected: str) -> None:
    """The verdict combines bioactivity clustering with structural similarity."""
    assert expected in chemistry.chemistry_verdict(p_value, tanimoto)
