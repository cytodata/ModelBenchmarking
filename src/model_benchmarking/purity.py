"""Measure how clean a canonical axis is in gene-expression space.

A useful axis should be driven by a small, coherent set of genes rather than a diffuse haze, and
should not simply track generic cell stress or the overall size of the response. For each axis
(correlated with every gene, as in :func:`model_benchmarking.decoding.gene_axis_statistics`) this
module computes

* **loading kurtosis**: how heavy-tailed the gene statistics are (high = a few genes dominate);
* **Hoyer sparsity**: 0 when all genes contribute equally, 1 when a single gene does;
* **pan-stress contamination**: the share of the 50 most positively correlated genes that belong to a
  generic stress/cytotoxicity panel;
* **total-signal correlation**: correlation of the axis with the sum of a compound's expression
  changes, a crude proxy for how strongly the compound perturbs the cells overall.
"""

from collections.abc import Collection, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, pearsonr

GENERIC_STRESS_MARKERS = frozenset(
    {"HMOX1", "DDIT3", "CDKN1A", "GADD45A", "ATF3", "ATF4", "HSPA1A", "HSPA1B", "FOS", "JUN", "EGR1", "IER3", "BBC3"}
)

VERDICT_COLORS = {"💎 Pristine": "#2A9D8F", "⚠️ Mixed / Moderate": "#E76F51", "❌ Stressed / Artifact": "#E63946"}


def hoyer_sparsity(values: np.ndarray) -> float:
    """Hoyer sparsity ``(sqrt(n) - L1/L2) / (sqrt(n) - 1)`` of a vector (0 = dense, 1 = one non-zero)."""
    n = len(values)
    l1 = np.sum(np.abs(values))
    l2 = np.linalg.norm(values)
    return float((np.sqrt(n) - l1 / (l2 + 1e-12)) / (np.sqrt(n) - 1))


def stress_contamination(
    correlations: np.ndarray,
    symbols: Sequence[str],
    markers: Collection[str] = GENERIC_STRESS_MARKERS,
    top_n: int = 50,
) -> tuple[float, set[str]]:
    """Percentage of the ``top_n`` most positively correlated genes that are generic stress markers.

    Parameters
    ----------
    correlations : numpy.ndarray, shape (n_genes,)
        Correlation of each gene with the axis.
    symbols : sequence of str
        Gene symbol of each gene (same order).
    markers : collection of str
        Stress-marker gene symbols.
    top_n : int, default 50

    Returns
    -------
    percent : float
    hits : set of str
        The marker genes found among the top genes.
    """
    top_idx = np.argsort(correlations)[-top_n:]
    hits = {symbols[i] for i in top_idx} & set(markers)
    return 100.0 * len(hits) / top_n, hits


def print_axis_purity(gene_df: pd.DataFrame, kurtosis_floor: float = 6.0, hoyer_floor: float = 0.40) -> None:
    """Print the purity profile of one axis.

    Parameters
    ----------
    gene_df : pandas.DataFrame
        Output of :func:`model_benchmarking.decoding.gene_axis_statistics` for the axis.
    kurtosis_floor, hoyer_floor : float
        Values above which the axis is called sparse.
    """
    t_vals = gene_df["T_Stat"].to_numpy()
    kurt = float(kurtosis(t_vals, fisher=True))
    hoyer = hoyer_sparsity(t_vals)
    stress_pct, stress_hits = stress_contamination(gene_df["Correlation"].to_numpy(), gene_df["Gene_Symbol"].tolist())

    print("\n  🧼 Transcriptional Purity Profile:")
    print(
        f"     • Loading Kurtosis (κ)      : {kurt:5.2f} (Target: > {kurtosis_floor}) "
        f"[{'✅ Sparse' if kurt > kurtosis_floor else '⚠️ Diffuse'}]"
    )
    print(
        f"     • Hoyer Sparsity Index      : {hoyer:5.3f} (Target: > {hoyer_floor}) "
        f"[{'✅ High' if hoyer > hoyer_floor else '⚠️ Dense'}]"
    )
    print(
        f"     • Pan-Stress Contamination  : {stress_pct:.1f}% ({len(stress_hits)}/50) "
        f"[{'✅ Clean' if stress_pct < 6.0 else '⚠️ Stressed'}]"
    )


def axis_purity_table(
    axis_scores: np.ndarray,
    axis_names: Sequence[str],
    gene_matrix: np.ndarray,
    symbols: Sequence[str],
    cell_counts: np.ndarray | None = None,
) -> pd.DataFrame:
    """Purity metrics of every axis, ranked by loading kurtosis and Hoyer sparsity.

    Parameters
    ----------
    axis_scores : numpy.ndarray, shape (n_compounds, n_axes)
        Compound scores on each axis.
    axis_names : sequence of str
        Name of each axis column.
    gene_matrix : numpy.ndarray, shape (n_compounds, n_genes)
        Gene expression of the same compounds, same row order.
    symbols : sequence of str
        Gene symbol of each gene column.
    cell_counts : numpy.ndarray, shape (n_compounds,), optional
        Cell count per compound, to test for a cell-number artifact.

    Returns
    -------
    pandas.DataFrame
        One row per axis with columns ``Axis``, ``Kurtosis (κ)``, ``Hoyer Sparsity``, ``Pan-Stress %``,
        ``Total-Signal |r|``, ``Cell Count |r|``, ``Top 3 Upregulated Genes`` and ``Quality Verdict``
        (pristine: κ ≥ 5, Hoyer ≥ 0.38, stress < 6 % and |r| with total signal < 0.25; artifact:
        stress ≥ 14 % or |r| with cell count > 0.45).

    Notes
    -----
    "Total signal" is the sum of a compound's gene values (log fold-changes), not sequencing depth.
    """
    Z = np.nan_to_num(np.asarray(axis_scores, dtype=np.float64))
    Y = np.nan_to_num(np.asarray(gene_matrix, dtype=np.float64))
    n_samples, n_genes = Y.shape

    Z_c = Z - Z.mean(axis=0, keepdims=True)
    Y_c = Y - Y.mean(axis=0, keepdims=True)
    norm_z = np.sqrt(np.sum(Z_c**2, axis=0, keepdims=True)).T
    norm_y = np.sqrt(np.sum(Y_c**2, axis=0, keepdims=True))

    # Pearson r of every axis with every gene in one matrix product
    R = (Z_c.T @ Y_c) / (norm_z * norm_y + 1e-12)
    R_clipped = np.clip(R, -0.9999, 0.9999)
    T = R_clipped * np.sqrt((n_samples - 2) / (1.0 - R_clipped**2))

    total_signal = Y.sum(axis=1)
    symbols_arr = np.array(symbols)

    records = []
    for i, name in enumerate(axis_names):
        kurt = float(kurtosis(T[i], fisher=True))
        hoyer = hoyer_sparsity(T[i])
        stress_pct, _ = stress_contamination(R[i], list(symbols))
        r_signal = abs(pearsonr(total_signal, Z[:, i])[0])
        r_count = abs(pearsonr(cell_counts, Z[:, i])[0]) if cell_counts is not None else np.nan

        stressed = stress_pct >= 14.0 or (not np.isnan(r_count) and r_count > 0.45)
        pristine = not stressed and kurt >= 5.0 and hoyer >= 0.38 and stress_pct < 6.0 and r_signal < 0.25
        verdict = "💎 Pristine" if pristine else "❌ Stressed / Artifact" if stressed else "⚠️ Mixed / Moderate"

        records.append(
            {
                "Axis": name,
                "Kurtosis (κ)": kurt,
                "Hoyer Sparsity": hoyer,
                "Pan-Stress %": stress_pct,
                "Total-Signal |r|": r_signal,
                "Cell Count |r|": None if np.isnan(r_count) else r_count,
                "Top 3 Upregulated Genes": ", ".join(symbols_arr[np.argsort(R[i])[-3:][::-1]]),
                "Quality Verdict": verdict,
            }
        )

    return (
        pd.DataFrame(records).sort_values(by=["Kurtosis (κ)", "Hoyer Sparsity"], ascending=False).reset_index(drop=True)
    )


def plot_purity_sweep(purity_df: pd.DataFrame) -> None:
    """Two-panel overview of :func:`axis_purity_table`: sparsity landscape and stress contamination."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5), dpi=130)

    for verdict, color in VERDICT_COLORS.items():
        sub = purity_df[purity_df["Quality Verdict"] == verdict]
        if len(sub) > 0:
            ax1.scatter(
                sub["Hoyer Sparsity"],
                sub["Kurtosis (κ)"],
                color=color,
                s=80,
                label=verdict,
                edgecolors="black",
                linewidth=0.8,
                alpha=0.85,
            )

    for _, row in purity_df.iterrows():
        ax1.annotate(
            row["Axis"].replace("_Axis_", "_"),
            (row["Hoyer Sparsity"], row["Kurtosis (κ)"]),
            fontsize=8,
            fontweight="bold",
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax1.axhline(y=6.0, color="gray", linestyle="--", linewidth=0.9, alpha=0.6, label="Kurtosis Floor (6.0)")
    ax1.axvline(x=0.40, color="gray", linestyle=":", linewidth=0.9, alpha=0.6, label="Hoyer Floor (0.40)")
    ax1.set_title("Transcriptional Sparsity & Cleanliness Landscape", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Hoyer Sparsity Index (0 = Dense, 1 = Ultra-Sparse)", fontsize=10)
    ax1.set_ylabel("Loading Kurtosis (κ)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8.5, frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.bar(
        purity_df["Axis"],
        purity_df["Pan-Stress %"],
        color=[VERDICT_COLORS.get(v, "#2B2D42") for v in purity_df["Quality Verdict"]],
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
    )
    ax2.axhline(y=6.0, color="#2A9D8F", linestyle="--", linewidth=1.2, label="Clean Target Threshold (<6%)")
    ax2.axhline(y=14.0, color="#E63946", linestyle="--", linewidth=1.2, label="Cytotoxic Artifact Threshold (>14%)")
    ax2.set_title("Generic Stress Marker Contamination (Pan-Cytotoxicity)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("% of Top 50 Genes in Pan-Stress Panel", fontsize=10)
    ax2.set_ylim(0, max(25.0, purity_df["Pan-Stress %"].max() + 5.0))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax2.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()
