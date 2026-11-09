"""Tests for model_benchmarking.purity (sparsity, kurtosis and stress-contamination diagnostics)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from model_benchmarking import purity


def test_hoyer_sparsity_extremes() -> None:
    """A single non-zero entry is maximally sparse; equal entries are maximally dense."""
    assert purity.hoyer_sparsity(np.array([0.0, 0.0, 5.0, 0.0])) == pytest.approx(1.0)
    assert purity.hoyer_sparsity(np.ones(16)) == pytest.approx(0.0, abs=1e-9)


def test_stress_contamination_counts_markers_among_top_genes() -> None:
    """Only markers among the most positively correlated genes count."""
    symbols = ["FOS", "JUN", "GENE1", "GENE2", "ATF3"]
    corr = np.array([0.9, 0.8, 0.1, 0.0, -0.9])  # ATF3 is strongly *negative*
    pct, hits = purity.stress_contamination(corr, symbols, top_n=2)
    assert hits == {"FOS", "JUN"} and pct == pytest.approx(100.0)
    pct, hits = purity.stress_contamination(corr, symbols, top_n=3)
    assert hits == {"FOS", "JUN"} and pct == pytest.approx(100 * 2 / 3)


@pytest.fixture
def sweep_data() -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Two axes: one driven by two private genes (pure), one tracking stress genes (artifact)."""
    rng = np.random.default_rng(0)
    n, g = 300, 120
    genes = rng.normal(size=(n, g))
    axes = rng.normal(size=(n, 2))
    genes[:, 0] += 3 * axes[:, 0]
    genes[:, 1] += 3 * axes[:, 0]
    stress_cols = list(range(2, 10))
    for c in stress_cols:
        genes[:, c] += 3 * axes[:, 1]
    symbols = (
        ["PURE1", "PURE2"]
        + ["FOS", "JUN", "ATF3", "ATF4", "EGR1", "IER3", "BBC3", "GADD45A"]
        + [f"G{i}" for i in range(g - 10)]
    )
    return axes, ["Axis_1", "Axis_2"], genes, symbols


def test_axis_purity_table(sweep_data: tuple[np.ndarray, list[str], np.ndarray, list[str]]) -> None:
    """The stress-driven axis is flagged as an artifact and the sparse axis ranks first."""
    axes, names, genes, symbols = sweep_data
    table = purity.axis_purity_table(axes, names, genes, symbols)
    assert table["Axis"].tolist() == ["Axis_1", "Axis_2"]
    by_axis = table.set_index("Axis")
    assert by_axis.loc["Axis_2", "Quality Verdict"] == "❌ Stressed / Artifact"
    assert by_axis.loc["Axis_1", "Pan-Stress %"] < by_axis.loc["Axis_2", "Pan-Stress %"]
    assert by_axis.loc["Axis_1", "Top 3 Upregulated Genes"].startswith(("PURE1", "PURE2"))
    assert by_axis["Cell Count |r|"].isna().all()


def test_axis_purity_table_flags_cell_count_artifact(
    sweep_data: tuple[np.ndarray, list[str], np.ndarray, list[str]],
) -> None:
    """An axis that tracks the cell count is flagged even without stress genes."""
    axes, names, genes, symbols = sweep_data
    table = purity.axis_purity_table(axes, names, genes, symbols, cell_counts=axes[:, 0])
    by_axis = table.set_index("Axis")
    assert by_axis.loc["Axis_1", "Quality Verdict"] == "❌ Stressed / Artifact"
    assert by_axis.loc["Axis_1", "Cell Count |r|"] == pytest.approx(1.0)


def test_print_axis_purity(capsys: pytest.CaptureFixture) -> None:
    """The purity profile prints the three metrics."""
    rng = np.random.default_rng(0)
    gene_df = pd.DataFrame(
        {
            "T_Stat": rng.normal(size=200),
            "Correlation": rng.normal(size=200),
            "Gene_Symbol": [f"G{i}" for i in range(200)],
        }
    )
    purity.print_axis_purity(gene_df)
    out = capsys.readouterr().out
    assert "Loading Kurtosis" in out and "Hoyer Sparsity" in out and "Pan-Stress Contamination  : 0.0%" in out


def test_plot_purity_sweep(sweep_data: tuple[np.ndarray, list[str], np.ndarray, list[str]]) -> None:
    """The sweep figure has the two expected panels."""
    axes, names, genes, symbols = sweep_data
    purity.plot_purity_sweep(purity.axis_purity_table(axes, names, genes, symbols))
    assert len(plt.gcf().axes) == 2
