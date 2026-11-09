"""Tests for model_benchmarking.pooling, .data and .neighborhoods."""

import numpy as np
import pandas as pd
import pytest
import torch

from model_benchmarking.data import safely_set_inchikey_index
from model_benchmarking.neighborhoods import evaluate_driver_neighborhoods, get_pairwise_sims
from model_benchmarking.pooling import max_abs_pool, max_absolute_pooling

CPU = torch.device("cpu")
KEY_A = "IAZDPXIOMUYVGZ-UHFFFAOYSA-N"
KEY_B = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"


def test_max_abs_pool_keeps_signed_extreme() -> None:
    """Max-absolute pooling keeps, per feature, the largest-magnitude value with its sign."""
    x = pd.DataFrame({"g1": [0.5, -2.0, 1.0], "g2": [0.1, 0.2, 3.0]})
    np.testing.assert_array_equal(max_abs_pool(x), [-2.0, 3.0])


def test_max_absolute_pooling_one_row_per_group() -> None:
    """Pooling a long table returns one row per group with the group column first."""
    df = pd.DataFrame(
        {
            "InChIKey": [KEY_A, KEY_A, KEY_B],
            "g1": [1.0, -4.0, 2.0],
            "g2": [0.0, 1.0, -5.0],
        }
    )
    out = max_absolute_pooling(df, "InChIKey", ["g1", "g2"])
    assert list(out.columns) == ["InChIKey", "g1", "g2"]
    assert out.set_index("InChIKey").loc[KEY_A].tolist() == [-4.0, 1.0]
    assert out.set_index("InChIKey").loc[KEY_B].tolist() == [2.0, -5.0]


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"Metadata_InChIKey": [KEY_A, KEY_B], "v": [1.0, 2.0]}),
        pd.DataFrame({"v": [1.0, 2.0]}, index=pd.Index([KEY_A, KEY_B], name="inchikey")),
        pd.DataFrame({"v": [1.0, 2.0]}, index=[KEY_A, KEY_B]),
    ],
)
def test_safely_set_inchikey_index(frame: pd.DataFrame) -> None:
    """InChIKeys become the index whether they start as a column, a named index or an unnamed index."""
    out = safely_set_inchikey_index(frame, "test")
    assert out.index.name == "InChIKey"
    assert list(out.index) == [KEY_A, KEY_B]
    assert list(out.columns) == ["v"]


def test_safely_set_inchikey_index_raises_without_key() -> None:
    """A table without any InChIKey column or index raises a KeyError."""
    with pytest.raises(KeyError):
        safely_set_inchikey_index(pd.DataFrame({"v": [1.0]}, index=[0]), "test")


def test_pairwise_sims() -> None:
    """Pairwise cosine similarities are returned for every unordered pair of rows."""
    t = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(get_pairwise_sims(t), [0.0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-6)


def test_driver_neighborhoods_detects_planted_cluster() -> None:
    """A planted cluster of drivers is strongly enriched in its space but not in a random one."""
    rng = np.random.default_rng(0)
    n, n_drivers = 200, 15
    keys = np.array([f"K{i:03d}" for i in range(n)])
    clustered = rng.normal(size=(n, 16))
    clustered[:n_drivers] = rng.normal(size=16) + 0.1 * rng.normal(size=(n_drivers, 16))
    random_space = rng.normal(size=(n, 16))

    out = evaluate_driver_neighborhoods(
        keys[:n_drivers],
        keys,
        {"clustered": clustered, "random": random_space},
        k_neighbors=5,
        device=CPU,
    ).set_index("Representation Space")

    assert out.loc["clustered", "Fold Enrichment"] > 10
    assert out.loc["clustered", "Hypergeom p-val"] < 1e-10
    assert out.loc["clustered", "mAP Score"] > 0.8
    assert out.loc["random", "Fold Enrichment"] < 3
