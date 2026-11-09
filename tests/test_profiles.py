"""Tests for model_benchmarking.profiles (caching, well selection and aggregation), without cloud access."""

import subprocess
from collections.abc import Iterator
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from model_benchmarking import profiles


@pytest.fixture
def conn() -> Iterator[duckdb.DuckDBPyConnection]:
    """An in-memory DuckDB connection."""
    connection = duckdb.connect()
    yield connection
    connection.close()


def test_cache_filename() -> None:
    """Model names are turned into safe cache file names."""
    assert profiles.cache_filename("CP CNN drugseq subset (normalized, 512d)") == (
        "cached_CP_CNN_drugseq_subset_normalized_512d.parquet"
    )
    assert profiles.cache_filename("DINO (Normed + Actives)") == "cached_DINO_Normed_plus_Actives.parquet"


def test_cache_from_gcs_uses_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An existing cache file is used without calling gcloud."""
    cache = tmp_path / "c.parquet"
    cache.write_bytes(b"data")
    monkeypatch.setattr(profiles.subprocess, "run", lambda *a, **k: pytest.fail("gcloud should not run"))
    assert profiles.cache_from_gcs("gs://b/x", str(cache), "model")


def test_cache_from_gcs_replaces_empty_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty leftover file triggers a fresh download."""
    cache = tmp_path / "c.parquet"
    cache.write_bytes(b"")

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        Path(cmd[-1]).write_bytes(b"x" * 10)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(profiles.subprocess, "run", fake_run)
    assert profiles.cache_from_gcs("gs://b/x", str(cache), "model")
    assert cache.stat().st_size == 10


def test_cache_from_gcs_reports_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A failed download returns False and prints gcloud's error message."""
    monkeypatch.setattr(profiles.subprocess, "run", lambda cmd, **k: subprocess.CompletedProcess(cmd, 1, "", "denied"))
    assert not profiles.cache_from_gcs("gs://b/x", str(tmp_path / "c.parquet"), "model")
    assert "Failed to download model: denied" in capsys.readouterr().out


def test_read_treated_wells(tmp_path: Path, conn: duckdb.DuckDBPyConnection) -> None:
    """Only treated wells of compounds in the key view are read, with the key column renamed."""
    wells = pd.DataFrame(
        {
            "Metadata_InChIKey": ["A", "A", "B", "C"],
            "Metadata_pert_type": ["trt", "control", "trt", "trt"],
            "f1": [1.0, 2.0, 3.0, 4.0],
        }
    )
    wells.to_parquet(tmp_path / "w.parquet")
    conn.register("keys_view", pd.DataFrame({"InChIKey": ["A", "B"]}))
    out = profiles.read_treated_wells(conn, str(tmp_path / "w.parquet"), "m", "keys_view")
    assert sorted(out["InChIKey"]) == ["A", "B"] and sorted(out["f1"]) == [1.0, 3.0]


def test_read_treated_wells_problems(
    tmp_path: Path, conn: duckdb.DuckDBPyConnection, capsys: pytest.CaptureFixture
) -> None:
    """Unreadable files, tables without InChIKeys and empty selections return None with a message."""
    conn.register("keys_view", pd.DataFrame({"InChIKey": ["Z"]}))
    assert profiles.read_treated_wells(conn, str(tmp_path / "missing.parquet"), "m", "keys_view") is None
    pd.DataFrame({"f1": [1.0]}).to_parquet(tmp_path / "nokey.parquet")
    assert profiles.read_treated_wells(conn, str(tmp_path / "nokey.parquet"), "m", "keys_view") is None
    pd.DataFrame({"InChIKey": ["A"], "f1": [1.0]}).to_parquet(tmp_path / "w.parquet")
    assert profiles.read_treated_wells(conn, str(tmp_path / "w.parquet"), "m", "keys_view") is None
    out = capsys.readouterr().out
    assert "CORRUPT FILE" in out and "No InChIKey found" in out and "Zero overlapping compounds" in out


def test_group_and_feature_columns() -> None:
    """Dose and time columns become grouping columns and are excluded from the features."""
    df = pd.DataFrame(
        columns=["InChIKey", "concentration", "dose", "pert_time_h", "Metadata_x", "pubchem_cid", "f1", "f2"]
    ).astype(
        {"concentration": float, "dose": float, "pert_time_h": float, "pubchem_cid": float, "f1": float, "f2": float}
    )
    group_cols = profiles.detect_group_columns(df, "InChIKey")
    assert group_cols == ["InChIKey", "concentration", "pert_time_h"]
    assert profiles.morphology_feature_columns(df, "InChIKey", group_cols) == ["dose", "f1", "f2"]


def test_median_by_group_matches_pandas(conn: duckdb.DuckDBPyConnection) -> None:
    """DuckDB median aggregation equals pandas groupby-median."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"InChIKey": rng.choice(list("ABC"), 30), "f1": rng.normal(size=30), "f2": rng.normal(size=30)})
    out = profiles.median_by_group_duckdb(conn, df, ["InChIKey"], ["f1", "f2"])
    expected = df.groupby("InChIKey", as_index=False)[["f1", "f2"]].median()
    pd.testing.assert_frame_equal(out, expected, check_dtype=False)


def test_adjusted_svcca_auc() -> None:
    """The adjusted AUC is the mean true correlation minus the mean null over the first axes."""
    true_auc, null_auc, adj = profiles.adjusted_svcca_auc(np.array([0.9, 0.5, 0.1]), np.array([0.3, 0.2, 0.1]), 2)
    assert (true_auc, null_auc) == pytest.approx((0.7, 0.25))
    assert adj == pytest.approx(0.45)
