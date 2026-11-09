"""Tests for model_benchmarking.decoding (axis statistics, reports and volcano plot), without network."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from model_benchmarking import decoding

CPU = torch.device("cpu")


@pytest.fixture
def gene_data() -> tuple[np.ndarray, torch.Tensor, torch.Tensor, list[str]]:
    """An axis and three genes: one tracking the axis, one opposing it and one noise gene."""
    rng = np.random.default_rng(0)
    axis = rng.normal(size=100)
    genes = np.column_stack([axis, -axis + 0.1 * rng.normal(size=100), rng.normal(size=100)])
    Y = torch.tensor(genes)
    Y_c = Y - Y.mean(dim=0)
    return axis, Y_c, torch.sum(Y_c**2, dim=0), ["ENSG01.1", "ENSG02", "ENSG03"]


def test_gene_axis_statistics(gene_data: tuple[np.ndarray, torch.Tensor, torch.Tensor, list[str]]) -> None:
    """Genes following or opposing the axis get correlations near +1 and -1 with tiny p-values."""
    axis, Y_c, var, ids = gene_data
    df = decoding.gene_axis_statistics(axis, Y_c, var, ids, {"ENSG01": "UP", "ENSG02": "DOWN"}, device=CPU)
    assert df["Gene_Symbol"].tolist() == ["UP", "DOWN", "ENSG03"]
    assert df.loc[0, "Correlation"] == pytest.approx(1.0)
    assert df.loc[1, "Correlation"] < -0.99
    assert df.loc[0, "P_Value"] < 1e-10 and df.loc[2, "P_Value"] > 0.01


def test_rank_genes_deduplicates_symbols() -> None:
    """Ranking drops unmapped genes, keeps the strongest duplicate symbol and sorts high to low."""
    gene_df = pd.DataFrame(
        {
            "Gene_Symbol": ["A", "A", "B", "ENSG9", "C"],
            "Correlation": [0.1, -0.9, 0.5, 0.99, -0.2],
            "T_Stat": [1.0, -9.0, 5.0, 9.9, -2.0],
        }
    )
    ranked = decoding.rank_genes(gene_df, "T_Stat")
    assert ranked["Gene_Symbol"].tolist() == ["B", "C", "A"]
    assert ranked["T_Stat"].tolist() == [5.0, -2.0, -9.0]


def test_print_top_genes(capsys: pytest.CaptureFixture) -> None:
    """Top genes are printed with their signed correlations."""
    top = pd.DataFrame({"Gene_Symbol": ["X"], "Correlation": [0.5]})
    decoding.print_top_genes(top, top.assign(Correlation=-0.25))
    out = capsys.readouterr().out
    assert "Key Upregulated   : X (+0.50)" in out and "Key Downregulated : X (-0.25)" in out


def test_print_gsea_formats_pathways(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """GSEA results are split into positive and negative lists with the HALLMARK_ prefix removed."""
    res = pd.DataFrame({"Term": ["HALLMARK_P53", "HALLMARK_MYC"], "NES": [2.0, -1.5], "FDR q-val": [0.001, 0.2]})
    monkeypatch.setattr(decoding.gp, "prerank", lambda **kwargs: SimpleNamespace(res2d=res))
    decoding.print_gsea(pd.DataFrame(), "MSigDB_Hallmark_2020", "UP:", "DOWN:", term_width=10)
    out = capsys.readouterr().out
    assert "\n  UP:\n     • P53        | NES: +2.00 | FDR: 1.00e-03" in out
    assert "\n  DOWN:\n     • MYC        | NES: -1.50" in out


def test_print_gsea_reports_errors(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """A failing GSEA call is reported instead of raised."""

    def fail(**kwargs: object) -> None:
        raise RuntimeError("offline")

    monkeypatch.setattr(decoding.gp, "prerank", fail)
    decoding.print_gsea(pd.DataFrame(), "x", "UP:", "DOWN:", term_width=10)
    assert "GSEA note: offline" in capsys.readouterr().out


def test_print_ora(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Over-representation results for both ends of the ranking are printed."""
    res = pd.DataFrame({"Term": ["HALLMARK_E2F", "HALLMARK_G2M"], "Adjusted P-value": [0.01, 0.2]})
    monkeypatch.setattr(decoding.gp, "enrichr", lambda **kwargs: SimpleNamespace(res2d=res))
    decoding.print_ora(pd.DataFrame({"Gene_Symbol": list("ABC")}))
    out = capsys.readouterr().out
    assert "Upregulated Program   : E2F (p=1.0e-02), G2M (p=2.0e-01)" in out
    assert "Downregulated Program" in out


def test_report_driver_similarity_detects_cluster(capsys: pytest.CaptureFixture) -> None:
    """Drivers that are near-identical in the reference space earn the 'distinct cluster' verdict."""
    rng = np.random.default_rng(1)
    n = 200
    keys = [f"K{i}" for i in range(n)]
    scores = np.zeros(n)
    scores[:8] = 10.0
    # Positive embeddings keep the background similarity above zero (the fold change divides by it).
    ref = rng.random(size=(n, 6))
    ref[:8] = rng.random(size=6) + 0.01 * rng.random(size=(8, 6))
    reference_df = pd.DataFrame(ref, columns=[f"d{i}" for i in range(6)]).assign(InChIKey=keys)
    drivers = decoding.report_driver_similarity(
        pd.DataFrame({"InChIKey": keys, "Bio_Axis_1": scores}),
        "Bio_Axis_1",
        reference_df,
        [f"d{i}" for i in range(6)],
        "InChIKey",
        decoding.CHEMICAL_CHECKER_STYLE,
        device=CPU,
    )
    assert sorted(drivers["InChIKey"]) == sorted(keys[:8])
    out = capsys.readouterr().out
    assert decoding.CHEMICAL_CHECKER_STYLE.title in out
    assert decoding.CHEMICAL_CHECKER_STYLE.verdict_distinct in out


def test_print_driver_annotations_consensus(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """A mechanism shared by at least two driver compounds is reported as a consensus."""
    monkeypatch.setattr(decoding, "get_rich_compound_annotation", lambda key: (f"cmpd-{key}", "HDAC inhibitor"))
    drivers = pd.DataFrame({"InChIKey": ["K1", "K2", "K3"], "Bio_Axis_1": [3.0, 2.0, 1.0]})
    decoding.print_driver_annotations(drivers, "Bio_Axis_1", "InChIKey")
    out = capsys.readouterr().out
    assert "1. cmpd-K1" in out
    assert "CONSENSUS MECHANISM FOUND: 'HDAC inhibitor' (3/3 compounds share this)" in out


def test_plot_volcano(gene_data: tuple[np.ndarray, torch.Tensor, torch.Tensor, list[str]]) -> None:
    """The volcano plot is drawn with the requested title."""
    axis, Y_c, var, ids = gene_data
    df = decoding.gene_axis_statistics(axis, Y_c, var, ids, {}, device=CPU)
    decoding.plot_volcano(df, "Volcano test")
    assert plt.gca().get_title() == "Volcano test"


def test_select_driver_compounds_threshold_and_fallback() -> None:
    """Drivers score above mean + 2 SD; if fewer than 5 qualify, the top 15 are used."""
    n = 100
    keys = [f"K{i}" for i in range(n)]
    ref = pd.DataFrame({"InChIKey": keys, "d0": np.arange(n, dtype=float)})
    scores = np.zeros(n)
    scores[:8] = 10.0
    merged, drivers = decoding.select_driver_compounds(
        pd.DataFrame({"InChIKey": keys, "Axis": scores}), "Axis", ref, "InChIKey"
    )
    assert len(merged) == n and sorted(drivers["InChIKey"]) == sorted(keys[:8])

    scores = np.linspace(0, 1, n)  # nothing exceeds mean + 2 SD
    _, drivers = decoding.select_driver_compounds(
        pd.DataFrame({"InChIKey": keys, "Axis": scores}), "Axis", ref, "InChIKey"
    )
    assert len(drivers) == 15 and drivers["InChIKey"].iloc[0] == "K99"


def test_report_driver_chemistry(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Tight drivers in bioactivity space with dissimilar scaffolds give the convergence verdict."""
    monkeypatch.setattr(decoding, "driver_tanimoto", lambda keys, path: (0.1, 0.1, 0.3, len(keys)))
    rng = np.random.default_rng(1)
    n = 200
    keys = [f"K{i}" for i in range(n)]
    scores = np.zeros(n)
    scores[:8] = 10.0
    ref = rng.random(size=(n, 6))
    ref[:8] = rng.random(size=6) + 0.01 * rng.random(size=(8, 6))
    reference_df = pd.DataFrame(ref, columns=[f"d{i}" for i in range(6)]).assign(InChIKey=keys)
    drivers = decoding.report_driver_chemistry(
        pd.DataFrame({"InChIKey": keys, "Axis": scores}),
        "Axis",
        reference_df,
        [f"d{i}" for i in range(6)],
        "InChIKey",
        "unused.parquet",
        device=CPU,
    )
    assert sorted(drivers["InChIKey"]) == sorted(keys[:8])
    out = capsys.readouterr().out
    assert "8 resolved to SMILES" in out and "Mean = 0.100" in out and "TRUE PHENOTYPIC CONVERGENCE" in out


def test_report_driver_chemistry_too_few_compounds(capsys: pytest.CaptureFixture) -> None:
    """With fewer than five shared compounds the report says so instead of testing."""
    keys = ["K1", "K2", "K3"]
    decoding.report_driver_chemistry(
        pd.DataFrame({"InChIKey": keys, "Axis": [1.0, 2.0, 3.0]}),
        "Axis",
        pd.DataFrame({"InChIKey": keys, "d0": [1.0, 2.0, 3.0]}),
        ["d0"],
        "InChIKey",
        "unused.parquet",
        device=CPU,
    )
    assert "Insufficient compound overlap" in capsys.readouterr().out


def test_centered_unit_vectors_remove_common_direction() -> None:
    """Rows have unit length, and a direction shared by every profile no longer makes them similar."""
    rng = np.random.default_rng(0)
    X = 10.0 + rng.normal(size=(50, 8))
    raw_sim = decoding.get_pairwise_sims(torch.tensor(X, dtype=torch.float32)).mean()
    normed = decoding.centered_unit_vectors(X, device=CPU)
    assert torch.allclose(normed.norm(dim=1), torch.ones(50), atol=1e-5)
    assert raw_sim > 0.9 and abs(decoding.get_pairwise_sims(normed).mean()) < 0.1


@pytest.fixture
def morphology() -> tuple[torch.Tensor, dict[str, int], list[str]]:
    """Centered unit profiles in which the first six compounds share a phenotype."""
    rng = np.random.default_rng(2)
    n = 120
    X = rng.normal(size=(n, 16))
    X[:6] += 4.0 * rng.normal(size=16)
    keys = [f"K{i}" for i in range(n)]
    return decoding.centered_unit_vectors(X, device=CPU), {k: i for i, k in enumerate(keys)}, keys


def test_report_morphology_coherence_detects_shared_phenotype(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    morphology: tuple[torch.Tensor, dict[str, int], list[str]],
) -> None:
    """Drivers that share a phenotype are far more similar than random groups; the medoid is one of them."""
    monkeypatch.setattr(decoding, "get_rich_compound_annotation", lambda key: ("MedoidName", "MedoidMOA"))
    normed, index, keys = morphology
    result = decoding.report_morphology_coherence(keys[:6], normed, index)
    assert result["mean_driver"] > 0.5 > result["mean_background"] + 0.2 and result["fold_enrichment"] > 1
    assert result["p_value"] < 1e-5 and "HIGH COHERENCE" in result["verdict"]
    assert result["medoid_key"] in keys[:6] and len(result["driver_sims"]) == 15
    assert "MedoidName (MedoidMOA)" in capsys.readouterr().out


def test_report_morphology_coherence_random_drivers_and_too_few(
    monkeypatch: pytest.MonkeyPatch, morphology: tuple[torch.Tensor, dict[str, int], list[str]]
) -> None:
    """Random drivers are not coherent; fewer than three drivers (or unknown keys) give None."""
    monkeypatch.setattr(decoding, "get_rich_compound_annotation", lambda key: ("n", "m"))
    normed, index, keys = morphology
    result = decoding.report_morphology_coherence(keys[50:60], normed, index)
    assert "HIGH COHERENCE" not in result["verdict"]
    assert decoding.report_morphology_coherence(keys[:2], normed, index) is None
    assert decoding.report_morphology_coherence(["x", "y", "z"], normed, index) is None


def test_report_morphology_coherence_is_reproducible(
    monkeypatch: pytest.MonkeyPatch, morphology: tuple[torch.Tensor, dict[str, int], list[str]]
) -> None:
    """The background draws are seeded."""
    monkeypatch.setattr(decoding, "get_rich_compound_annotation", lambda key: ("n", "m"))
    normed, index, keys = morphology
    first = decoding.report_morphology_coherence(keys[:6], normed, index)
    second = decoding.report_morphology_coherence(keys[:6], normed, index)
    np.testing.assert_array_equal(first["background_sims"], second["background_sims"])


def test_plot_morphology_coherence_and_placeholder(
    monkeypatch: pytest.MonkeyPatch, morphology: tuple[torch.Tensor, dict[str, int], list[str]]
) -> None:
    """The density panel is drawn from a result, and a placeholder message when there is none."""
    monkeypatch.setattr(decoding, "get_rich_compound_annotation", lambda key: ("n", "m"))
    normed, index, keys = morphology
    result = decoding.report_morphology_coherence(keys[:6], normed, index)
    _, (ax1, ax2) = plt.subplots(1, 2)
    decoding.plot_morphology_coherence(result, "Axis_1", ax1)
    decoding.plot_morphology_coherence(None, "Axis_1", ax2)
    assert "Centered Morphological QC" in ax1.get_title() and ax2.texts[0].get_text().startswith("Insufficient")


def test_plot_volcano_on_given_axes(gene_data: tuple[np.ndarray, torch.Tensor, torch.Tensor, list[str]]) -> None:
    """With an axes argument the volcano is drawn there and no new figure is opened."""
    axis, Y_c, var, ids = gene_data
    df = decoding.gene_axis_statistics(axis, Y_c, var, ids, {}, device=CPU)
    fig, ax = plt.subplots()
    decoding.plot_volcano(df, "Panel volcano", ax=ax)
    assert ax.get_title() == "Panel volcano" and plt.get_fignums() == [fig.number]


def test_plot_morphology_coherence_without_fold() -> None:
    """A non-positive background mean has no fold change; the title then omits it."""
    coherence = {
        "driver_sims": np.array([0.1, 0.2, 0.3]),
        "background_sims": np.array([-0.2, -0.1, 0.0, 0.05]),
        "mean_driver": 0.2,
        "mean_background": -0.06,
        "fold_enrichment": float("nan"),
        "p_value": 0.01,
    }
    _, ax = plt.subplots()
    decoding.plot_morphology_coherence(coherence, "Axis_1", ax)
    assert "Fold" not in ax.get_title() and "p = 1.00e-02" in ax.get_title()
