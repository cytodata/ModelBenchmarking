"""Tests for model_benchmarking.anchors (identifier detection, cache validation and overlap loading)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from model_benchmarking import anchors

DMSO = anchors.DMSO_INCHIKEY


def write_parquet(path: Path, df: pd.DataFrame, **kwargs: object) -> str:
    """Write ``df`` to ``path`` and return the path as a string."""
    df.to_parquet(path, **kwargs)
    return str(path)


@pytest.fixture
def datasets(tmp_path: Path) -> tuple[str, str]:
    """A morphology table with replicate wells (and DMSO/control rows) and a DRUG-seq table with several doses."""
    morph = pd.DataFrame(
        {
            "Metadata_InChIKey": ["B", "B", "A", "A", "A", DMSO, "C", "A"],
            "Metadata_pert_type": ["trt", "trt", "trt", "trt", "trt", "trt", "trt", "ctl"],
            "Metadata_Plate": list("pqrstuvw"),
            "emb_0": [1.0, 3.0, 10.0, 20.0, 60.0, 0.0, 5.0, 999.0],
            "emb_1": [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 2.0, 999.0],
        }
    )
    rna = pd.DataFrame(
        {
            "InChIKey": ["A", "A", "B", "D", DMSO],
            "pert_time_h": [24, 48, 24, 24, 24],
            "pert_dose_uM": [10.0, 10.0, 10.0, 10.0, 0.0],
            "g1": [1.0, -3.0, 2.0, 7.0, 0.0],
            "g2": [0.5, 0.2, -1.0, 7.0, 0.0],
        }
    )
    return write_parquet(tmp_path / "morph.parquet", morph), write_parquet(tmp_path / "rna.parquet", rna)


def test_detect_inchikey_col(tmp_path: Path) -> None:
    """The InChIKey column is found by name, skipping truncated 14-character keys."""
    path = write_parquet(
        tmp_path / "a.parquet", pd.DataFrame({"InChIKey14": ["x"], "Metadata_InChIKey": ["y"], "v": [1]})
    )
    assert anchors.detect_inchikey_col(path) == "Metadata_InChIKey"
    assert anchors.detect_inchikey_col(write_parquet(tmp_path / "b.parquet", pd.DataFrame({"v": [1]}))) is None


def test_detect_inchikey_col_from_pandas_index(tmp_path: Path) -> None:
    """A key stored only as the pandas index is found through the Parquet metadata."""
    df = pd.DataFrame({"v": [1, 2]}, index=pd.Index(["a", "b"], name="compound_inchikey"))
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(table, tmp_path / "c.parquet")
    assert anchors.detect_inchikey_col(str(tmp_path / "c.parquet")) == "compound_inchikey"


def test_resolve_key_column_prefers_known_names() -> None:
    """Exact known names win over other columns that merely mention 'inchikey'."""
    assert anchors.resolve_key_column(["x_inchikey", "InChIKey"]) == "InChIKey"
    assert anchors.resolve_key_column(["a", "my_inchikey_col"]) == "my_inchikey_col"
    assert anchors.resolve_key_column(["a"]) is None


def test_standardize_inchikey_column() -> None:
    """The key is renamed, redundant key columns are dropped, and a missing key is an error."""
    df = pd.DataFrame({"Metadata_InChIKey_x": ["a"], "Metadata_InChIKey_y": ["a"], "v": [1.0]})
    out = anchors.standardize_inchikey_column(df, "InChIKey")
    assert list(out.columns) == ["InChIKey", "v"] or list(out.columns) == ["v", "InChIKey"]
    with pytest.raises(ValueError, match="No InChIKey"):
        anchors.standardize_inchikey_column(pd.DataFrame({"v": [1]}))


def test_cache_validated_from_gcs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Valid caches are kept; empty or key-less caches are deleted and downloaded again."""
    calls = []

    def fake_run(cmd: list[str], check: bool) -> None:
        calls.append(cmd)
        pd.DataFrame({"InChIKey": ["x"]}).to_parquet(cmd[-1])

    monkeypatch.setattr(anchors.subprocess, "run", fake_run)

    good = write_parquet(tmp_path / "good.parquet", pd.DataFrame({"InChIKey": ["x"]}))
    anchors.cache_validated_from_gcs("gs://b/good", good, "good")
    assert calls == []

    stale = write_parquet(tmp_path / "stale.parquet", pd.DataFrame({"v": [1]}))
    anchors.cache_validated_from_gcs("gs://b/stale", stale, "stale")
    empty = tmp_path / "empty.parquet"
    empty.write_bytes(b"")
    anchors.cache_validated_from_gcs("gs://b/empty", str(empty), "empty")
    missing = str(tmp_path / "missing.parquet")
    anchors.cache_validated_from_gcs("gs://b/missing", missing, "missing")
    assert [c[4] for c in calls] == [stale, str(empty), missing]
    assert anchors.detect_inchikey_col(stale) == "InChIKey"


def test_load_overlap_pair(datasets: tuple[str, str]) -> None:
    """Shared compounds are returned sorted; wells use the median, doses the max-absolute value; controls are dropped."""
    morph_path, rna_path = datasets
    X, Y, keys = anchors.load_overlap_pair(morph_path, rna_path)
    assert keys.tolist() == ["A", "B"]
    np.testing.assert_allclose(X, [[20.0, 1.0], [2.0, 0.0]])
    np.testing.assert_allclose(Y, [[-3.0, 0.5], [2.0, -1.0]])
    assert X.dtype == Y.dtype == np.float32


def test_load_overlap_pair_restricts_to_target_keys(datasets: tuple[str, str]) -> None:
    """A common cohort restricts the comparison to the requested compounds."""
    morph_path, rna_path = datasets
    _, _, keys = anchors.load_overlap_pair(morph_path, rna_path, target_keys={"B"})
    assert keys.tolist() == ["B"]


def test_load_overlap_pair_without_overlap(tmp_path: Path) -> None:
    """Datasets sharing no compound raise a clear error."""
    morph = write_parquet(tmp_path / "m.parquet", pd.DataFrame({"InChIKey": ["A"], "e": [1.0]}))
    rna = write_parquet(tmp_path / "r.parquet", pd.DataFrame({"InChIKey": ["B"], "g": [1.0]}))
    with pytest.raises(ValueError, match="Zero overlapping"):
        anchors.load_overlap_pair(morph, rna)


def test_load_overlap_pair_requires_identifier(tmp_path: Path) -> None:
    """A table without an InChIKey column cannot be paired."""
    morph = write_parquet(tmp_path / "m.parquet", pd.DataFrame({"e": [1.0]}))
    rna = write_parquet(tmp_path / "r.parquet", pd.DataFrame({"InChIKey": ["B"], "g": [1.0]}))
    with pytest.raises(ValueError, match="InChIKey column missing"):
        anchors.load_overlap_pair(morph, rna)


def test_find_common_compounds(datasets: tuple[str, str], tmp_path: Path) -> None:
    """The common cohort is the intersection of every morphology file with the anchor."""
    morph_path, rna_path = datasets
    other = write_parquet(
        tmp_path / "other.parquet", pd.DataFrame({"InChIKey": ["B", "Z", DMSO], "e": [1.0, 2.0, 3.0]})
    )
    assert anchors.find_common_compounds({"first": morph_path}, rna_path) == {"A", "B"}
    assert anchors.find_common_compounds({"first": morph_path, "other": other}, rna_path) == {"B"}


def test_metadata_json_is_valid_for_pandas_schema(tmp_path: Path) -> None:
    """Sanity check of the fixture helper used above: Parquet pandas metadata is JSON."""
    path = write_parquet(tmp_path / "x.parquet", pd.DataFrame({"a": [1]}))
    assert json.loads(pq.read_schema(path).metadata[b"pandas"])["columns"]


def test_cache_name_for_uri_separates_sources() -> None:
    """Files with the same name in different folders get different caches."""
    a = anchors.cache_name_for_uri("gs://bucket/embeddings/x/data.parquet")
    b = anchors.cache_name_for_uri("gs://bucket/indexed_by_inchikey/data.parquet")
    assert a == "cached_x__data.parquet" and b == "cached_indexed_by_inchikey__data.parquet" and a != b
