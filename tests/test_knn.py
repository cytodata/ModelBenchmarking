"""Tests for model_benchmarking.knn (kNN graphs, edge overlap statistics, table loading)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_benchmarking.knn import (
    edge_overlap_statistics,
    knn_neighbor_indices,
    load_embedding_tables,
    undirected_knn_edges,
)

KEYS = [f"{c * 14}-{c * 10}-N" for c in "ABCDEFGH"]


def test_knn_indices_list_each_point_first() -> None:
    """Each row's nearest neighbour under cosine distance is the row itself."""
    X = np.random.default_rng(0).normal(size=(30, 5))
    indices = knn_neighbor_indices(X, max_k=4)
    assert indices.shape == (30, 5)
    np.testing.assert_array_equal(indices[:, 0], np.arange(30))


def test_undirected_edges_are_deduplicated() -> None:
    """Mutual neighbours produce a single undirected edge and self-links are dropped."""
    indices = np.array([[0, 1, 2], [1, 0, 2], [2, 2, 0]])
    assert undirected_knn_edges(indices, k=1) == {(0, 1)}
    assert undirected_knn_edges(indices, k=2) == {(0, 1), (0, 2), (1, 2)}


def test_edge_overlap_identical_graphs() -> None:
    """Identical graphs overlap fully, with fold enrichment equal to one over the graph density."""
    edges = {(0, 1), (1, 2), (2, 3)}
    stats = edge_overlap_statistics(edges, edges, n_compounds=10)
    assert stats["Overlap_Pct"] == 100
    assert stats["Random_Overlap"] == pytest.approx(100 * 3 / 45)
    assert stats["Fold_Enrichment"] == pytest.approx(15)
    assert stats["P_Value"] < 0.01


def test_edge_overlap_disjoint_graphs() -> None:
    """Graphs without shared edges have zero overlap and a p-value of one."""
    stats = edge_overlap_statistics({(0, 1)}, {(2, 3)}, n_compounds=5)
    assert stats["Overlap_Pct"] == 0 and stats["Fold_Enrichment"] == 0
    assert stats["P_Value"] == 1.0 and stats["Likelihood_GT_Chance"] == "0.00%"


def _write_table(path: Path, n_rows: int, duplicate_first: bool = False) -> None:
    """Write a small embedding table with an InChIKey column, one text column and two features."""
    keys = KEYS[:n_rows]
    if duplicate_first:
        keys = [keys[0], *keys]
    df = pd.DataFrame(
        {
            "Metadata_InChIKey": keys,
            "Metadata_plate": ["p"] * len(keys),
            "f1": np.arange(len(keys), dtype=float),
            "f2": 1.0,
        }
    )
    df.to_parquet(path)


def test_load_embedding_tables_keeps_first_row_and_numeric_columns(tmp_path: Path) -> None:
    """Tables are indexed by InChIKey, keep the first of duplicate rows and only numeric columns."""
    _write_table(tmp_path / "a.parquet", 4, duplicate_first=True)
    out = load_embedding_tables({"A": str(tmp_path / "a.parquet")})
    df = out["A"]
    assert df.index.name == "InChIKey" and list(df.columns) == ["f1", "f2"]
    assert df.dtypes.unique().tolist() == [np.float32]
    assert len(df) == 4 and df.loc[KEYS[0], "f1"] == 0.0


def test_load_embedding_tables_filters_and_reports(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """With target keys, tables are subset, empty ones are skipped, and unreadable files are reported."""
    _write_table(tmp_path / "a.parquet", 4)
    config = {"A": str(tmp_path / "a.parquet"), "missing": str(tmp_path / "none.parquet")}
    out = load_embedding_tables(config, target_keys={KEYS[1], KEYS[2]})
    assert sorted(out["A"].index) == sorted([KEYS[1], KEYS[2]])
    assert "Failed to load missing" in capsys.readouterr().out

    out = load_embedding_tables({"A": str(tmp_path / "a.parquet")}, target_keys={"not-a-key"})
    assert out == {}
    assert "resulted in 0 matching molecules" in capsys.readouterr().out
