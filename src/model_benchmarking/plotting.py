"""Figures used by the nearest-neighbour (Step 3) and SVCCA pilot (Step 4) notebooks.

Every function draws one figure and shows it with ``plt.show()``.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.decomposition import PCA

# Line colour per image-model family (matched by name prefix) in the SVCCA figures.
MODEL_COLORS = {
    "DINO": "#E63946",  # Red
    "OpenPhenom 384": "#1D3557",  # Navy
    "OpenPhenom 1920": "#2A9D8F",  # Teal
    "CP CNN": "#9B5DE5",  # Purple
    "Imagenet": "#E76F51",  # Coral
}

# Order of the kNN heatmap axes: structure -> bioactivity -> morphology.
HEATMAP_ORDER = [
    "Minimol",
    "CC_Sign2",
    "CC_Sign3",
    "DINO_Raw",
    "DINO_Norm",
    "DINO_Norm_Actives",
    "OP384_Raw",
    "OP384_Norm",
    "OP384_Norm_Actives",
    "OP1920_Raw",
    "OP1920_Norm",
    "OP1920_Norm_Actives",
]


def _comparisons_with(df: pd.DataFrame, anchor_name: str) -> pd.DataFrame:
    """Rows of ``df`` involving ``anchor_name``, with the *other* dataset in ``Comparison``."""
    df_anchor = df[(df["Dataset_1"] == anchor_name) | (df["Dataset_2"] == anchor_name)].copy()
    df_anchor["Comparison"] = np.where(
        df_anchor["Dataset_1"] == anchor_name, df_anchor["Dataset_2"], df_anchor["Dataset_1"]
    )
    return df_anchor


def plot_heatmap(df: pd.DataFrame, metric: str, title: str, dataset_names: Sequence[str], k_target: int = 10) -> None:
    """Lower-triangle heatmap of a pairwise kNN metric between all datasets.

    Parameters
    ----------
    df : pandas.DataFrame
        Pairwise results with columns ``Dataset_1``, ``Dataset_2``, ``k`` and ``metric``.
    metric : str
        Column to plot (e.g. ``'Fold_Enrichment'`` or ``'Overlap_Pct'``).
    title : str
        Figure title.
    dataset_names : sequence of str
        All datasets that were compared (used to fill the diagonal).
    k_target : int, default 10
        Neighbourhood size to plot.

    Notes
    -----
    The table is mirrored to fill a square matrix, which assumes the metric is
    symmetric; the diagonal is set to the maximum value and hidden by the mask.
    """
    # Isolate the specific K
    df_k = df[df["k"] == k_target].copy()

    # We need to mirror the dataset to make a full square matrix
    df_mirror = df_k.rename(columns={"Dataset_1": "Dataset_2", "Dataset_2": "Dataset_1"})
    df_full = pd.concat([df_k, df_mirror])

    # Add self-comparisons as perfect scores on the diagonal for aesthetics
    self_comps = pd.DataFrame(
        {
            "Dataset_1": dataset_names,
            "Dataset_2": dataset_names,
            metric: [df_full[metric].max()] * len(dataset_names),  # Max out the diagonal
        }
    )
    df_full = pd.concat([df_full, self_comps])

    # Pivot into a square grid
    pivot_table = df_full.pivot(index="Dataset_1", columns="Dataset_2", values=metric)

    # Filter to only datasets that actually successfully joined
    ordered_cols = [c for c in HEATMAP_ORDER if c in pivot_table.columns]
    pivot_table = pivot_table.reindex(index=ordered_cols, columns=ordered_cols)

    plt.figure(figsize=(14, 11))

    # Mask the upper triangle for a cleaner look (optional, but standard for correlation matrices)
    mask = np.triu(np.ones_like(pivot_table, dtype=bool))

    sns.heatmap(
        pivot_table,
        mask=mask,
        cmap="viridis" if "Enrichment" in metric else "mako",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": metric},
    )

    plt.title(title, fontsize=18, pad=20)
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_anchor_sweep(
    df: pd.DataFrame,
    anchor_name: str,
    k_list: Sequence[int],
    palette: Any,
    metric: str = "Fold_Enrichment",
    metric_label: str = "Fold Enrichment",
) -> None:
    """One line per dataset compared with ``anchor_name``: ``metric`` as a function of k.

    Parameters
    ----------
    df : pandas.DataFrame
        Pairwise results (``Dataset_1``, ``Dataset_2``, ``k`` and ``metric`` columns).
    anchor_name : str
        Dataset whose comparisons are plotted.
    k_list : sequence of int
        Neighbourhood sizes (x-axis ticks).
    palette : seaborn palette
        Line colours.
    metric, metric_label : str
        Column to plot and its axis label.
    """
    df_anchor = _comparisons_with(df, anchor_name)
    if df_anchor.empty:
        print(f"⚠️ No successful alignments found for {anchor_name}.")
        return

    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df_anchor, x="k", y=metric, hue="Comparison", marker="o", linewidth=2.5, palette=palette)

    plt.xscale("log")
    plt.xticks(k_list)
    plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter())

    plt.title(f"k-Sweep: Comparing {anchor_name} vs All Phenotypic Spaces", fontsize=16, pad=15)
    plt.xlabel("Number of Neighbors (k)", fontsize=14)
    plt.ylabel(metric_label, fontsize=14)

    # Move legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Dataset Evaluated")
    plt.grid(True, alpha=0.3)
    sns.despine()
    plt.tight_layout()
    plt.show()


def _plot_overlap_and_metric(
    df_anchor: pd.DataFrame,
    anchor_name: str,
    k_list: Sequence[int],
    palette: Any,
    metric: str,
    right_title: str,
    right_ylabel: str,
    legend_title: str,
) -> None:
    """Two panels: neighbour overlap (%) on the left, ``metric`` on the right, versus k."""
    _, axes = plt.subplots(1, 2, figsize=(18, 6))

    # --- Plot 1: Raw Overlap Percentage ---
    sns.lineplot(
        data=df_anchor,
        x="k",
        y="Overlap_Pct",
        hue="Comparison",
        marker="o",
        linewidth=2.5,
        palette=palette,
        ax=axes[0],
    )
    axes[0].set_xscale("log")
    axes[0].set_xticks(k_list)
    axes[0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[0].set_title(f"Neighborhood Overlap (%)\n{anchor_name} vs Phenotypic Models", fontsize=14)
    axes[0].set_xlabel("Number of Neighbors (k)", fontsize=12)
    axes[0].set_ylabel("Unique Edge Overlap (%)", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].get_legend().remove()

    # --- Plot 2: Fold Enrichment (or another metric) ---
    sns.lineplot(
        data=df_anchor,
        x="k",
        y=metric,
        hue="Comparison",
        marker="s",
        linewidth=2.5,
        palette=palette,
        ax=axes[1],
    )
    axes[1].set_xscale("log")
    axes[1].set_xticks(k_list)
    axes[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[1].set_title(f"{right_title}\n{anchor_name} vs Phenotypic Models", fontsize=14)
    axes[1].set_xlabel("Number of Neighbors (k)", fontsize=12)
    axes[1].set_ylabel(right_ylabel, fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", title=legend_title)

    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_dual_metrics(df: pd.DataFrame, anchor_name: str, k_list: Sequence[int], palette: Any) -> None:
    """Overlap (%) and fold enrichment versus k for every dataset compared with ``anchor_name``.

    Parameters
    ----------
    df : pandas.DataFrame
        Pairwise results with ``Overlap_Pct`` and ``Fold_Enrichment`` columns.
    anchor_name : str
        Dataset whose comparisons are plotted.
    k_list : sequence of int
        Neighbourhood sizes (x-axis ticks).
    palette : seaborn palette
        Line colours.
    """
    df_anchor = _comparisons_with(df, anchor_name)
    if df_anchor.empty:
        return
    _plot_overlap_and_metric(
        df_anchor,
        anchor_name,
        k_list,
        palette,
        metric="Fold_Enrichment",
        right_title="Fold Enrichment (vs Random)",
        right_ylabel="Fold Enrichment (x times better)",
        legend_title="Morphological Model",
    )


def plot_anchor_sweep_dual(
    df: pd.DataFrame,
    anchor_name: str,
    k_list: Sequence[int],
    palette: Any,
    metric: str = "Fold_Enrichment",
    metric_label: str = "Fold Enrichment",
) -> None:
    """Overlap (%) and ``metric`` versus k for every dataset compared with ``anchor_name``.

    Same figure as :func:`plot_dual_metrics`, with a configurable right-hand metric and a
    warning when the anchor has no comparisons.

    Parameters
    ----------
    df : pandas.DataFrame
        Pairwise results with ``Overlap_Pct`` and ``metric`` columns.
    anchor_name : str
        Dataset whose comparisons are plotted.
    k_list : sequence of int
        Neighbourhood sizes (x-axis ticks).
    palette : seaborn palette
        Line colours.
    metric, metric_label : str
        Right-hand column to plot and its label.
    """
    df_anchor = _comparisons_with(df, anchor_name)
    if df_anchor.empty:
        print(f"⚠️ No successful alignments found for {anchor_name}.")
        return
    _plot_overlap_and_metric(
        df_anchor,
        anchor_name,
        k_list,
        palette,
        metric=metric,
        right_title=f"{metric_label} (vs Random)",
        right_ylabel=f"{metric_label} (x times better)",
        legend_title="Dataset Evaluated",
    )


def _canonical_correlations(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Canonical correlations between the column spaces of ``A`` and ``B``, strongest first."""
    # Orthonormalize bases to prevent collinearity inflation
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    # The singular values of the cross-covariance are exactly the canonical correlations
    _, S, _ = np.linalg.svd(Qa.T @ Qb, full_matrices=False)
    return np.sort(S)[::-1]


def plot_svcca_decay(
    X: np.ndarray, Y: np.ndarray, n_components: int = 100, n_permutations: int = 15
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute and plot the SVCCA decay curve of two representations with a permutation null.

    Both representations are reduced to their top ``k`` principal components
    (``k = min(n_components, 0.8 * n_compounds, n_features)``); the canonical
    correlations between the two PCA spaces are compared with those obtained after
    shuffling the rows of ``Y``.

    Parameters
    ----------
    X, Y : numpy.ndarray
        Paired representations (same compounds, same row order).
    n_components : int, default 100
        Maximum number of principal components per representation.
    n_permutations : int, default 15
        Number of shuffles used to build the null. Uses NumPy's global random state.

    Returns
    -------
    true_corrs : numpy.ndarray
        Canonical correlations, strongest first.
    null_mean : numpy.ndarray
        Mean canonical correlation over shuffles, per rank.
    sig_axes_count : int
        Number of ranks where the true correlation exceeds the null mean by 2 SD.
    """
    # 1. Ensure n_components isn't larger than our sample size
    N = X.shape[0]
    # Keep components well below N to prevent mathematical 1.0 hallucinations
    max_k = min(int(N * 0.8), X.shape[1], Y.shape[1])
    k = min(n_components, max_k)

    print(f"📊 Calculating SVCCA Axis Decay across top {k} dimensions...")

    # 2. PCA Truncation (The Denoiser Step)
    pca_X = PCA(n_components=k)
    pca_Y = PCA(n_components=k)

    X_pca = pca_X.fit_transform(X)
    Y_pca = pca_Y.fit_transform(Y)

    true_corrs = _canonical_correlations(X_pca, Y_pca)

    # 4. Permutation Control (Null Hallucination Baseline)
    print("🎲 Establishing null baseline via permutation...")
    null_corrs_matrix = np.zeros((n_permutations, k))

    for i in range(n_permutations):
        # Shuffle transcriptomic rows to break biological mapping
        Y_shuffled = Y_pca[np.random.permutation(N), :]
        null_corrs_matrix[i, :] = _canonical_correlations(X_pca, Y_shuffled)

    null_mean = null_corrs_matrix.mean(axis=0)
    null_std = null_corrs_matrix.std(axis=0)

    # 5. Plotting Engine
    plt.figure(figsize=(10, 6), dpi=120)
    sns.set_style("whitegrid")

    axes = np.arange(1, k + 1)

    # Plot true signal vs null baseline
    plt.plot(axes, true_corrs, color="#E63946", linewidth=2.5, label="True Biological Alignment")
    plt.plot(axes, null_mean, color="#1D3557", linestyle="--", linewidth=2, label="Null Permutation Baseline")
    plt.fill_between(axes, null_mean - null_std, null_mean + null_std, color="#1D3557", alpha=0.2)

    # We define significant axes as those > 2 standard deviations above the null mean
    sig_mask = true_corrs > (null_mean + (2 * null_std))
    sig_axes_count = np.sum(sig_mask)

    plt.axvline(x=sig_axes_count, color="gray", linestyle=":", label=f"Significance Cutoff ({sig_axes_count} Axes)")

    # Aesthetic formatting
    plt.title("SVCCA Canonical Axis Decay: DINO Morphology vs DRUG-seq Transcriptomics", fontsize=14, pad=15)
    plt.xlabel("Canonical Axis Rank", fontsize=12)
    plt.ylabel("Canonical Correlation (Pearson $r$)", fontsize=12)
    plt.legend(loc="upper right", fontsize=11)
    plt.xlim(1, k)
    plt.ylim(0, 1.05)
    plt.tight_layout()

    plt.show()

    return true_corrs, null_mean, sig_axes_count


def _model_style(name: str) -> tuple[str, str, float, float]:
    """Colour, line style, line width and opacity of a model's decay curve."""
    c = next((col for k, col in MODEL_COLORS.items() if name.startswith(k)), "gray")
    ls = ":" if "Raw" in name else "-"
    lw = 2.2 if ("Normed" in name or "normalized" in name) else 1.5
    alpha = 1.0 if ("Normed" in name or "normalized" in name) else 0.65
    return c, ls, lw, alpha


def _add_dual_legends(
    ax: plt.Axes,
    model_lines: list[Line2D],
    models_title: str,
    null_label: str,
    band_label: str,
    controls_title: str,
) -> None:
    """Legend of model curves (top right) and of the null baseline / noise band (below it)."""
    legend_models = ax.legend(
        handles=model_lines,
        title=models_title,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        fontsize=9.5,
        title_fontsize=11,
        frameon=True,
        shadow=True,
    )
    ax.add_artist(legend_models)

    control_handles = [
        Line2D([0], [0], color="#1A1A1A", linestyle="--", linewidth=2.4, label=null_label),
        Patch(facecolor="black", alpha=0.18, edgecolor="black", linestyle="--", label=band_label),
    ]
    ax.legend(
        handles=control_handles,
        title=controls_title,
        bbox_to_anchor=(1.02, 0.35),
        loc="upper left",
        fontsize=9.5,
        title_fontsize=11,
        frameon=True,
        shadow=True,
    )


def plot_svcca_decay_overview(
    decay_curves: Mapping[str, np.ndarray],
    null_curves: Mapping[str, tuple[np.ndarray, np.ndarray]],
    winner_name: str,
    max_axes: int,
    title: str,
    models_legend_title: str = "🧬 Morphological Models",
    band_label: str = "Noise Floor (±2σ, p = 0.05 cutoff)",
    legend_band_label: str = "Noise Floor (±2σ, p = 0.05 cutoff)",
) -> None:
    """SVCCA decay curves of all models, drawn against the top model's permutation null.

    Parameters
    ----------
    decay_curves : mapping of str to numpy.ndarray
        Model name -> canonical correlations, strongest first.
    null_curves : mapping of str to (numpy.ndarray, numpy.ndarray)
        Model name -> (null mean, null SD) per rank.
    winner_name : str
        Model whose null baseline is drawn (normally the top of the leaderboard).
    max_axes : int
        Maximum number of canonical axes shown.
    title : str
        Figure title.
    models_legend_title, band_label, legend_band_label : str
        Legend texts (the noise band's own label and its entry in the control legend).
    """
    winner_null_mean, winner_null_std = null_curves[winner_name]

    print(f"\nGenerating SVCCA Axis Decay Plot (Top Model: [{winner_name}])...")
    _, ax = plt.subplots(figsize=(14.5, 8), dpi=140)
    sns.set_style("whitegrid")

    model_lines = []
    max_available_dims = max([len(corrs) for corrs in decay_curves.values()])
    plot_horizon = min(max_axes, max_available_dims)

    for name, corrs in decay_curves.items():
        c, ls, lw, alpha = _model_style(name)
        plot_len = min(plot_horizon, len(corrs))
        (line,) = ax.plot(
            range(1, plot_len + 1), corrs[:plot_len], label=name, color=c, linestyle=ls, linewidth=lw, alpha=alpha
        )
        model_lines.append(line)

    # Winner null baseline and 95% noise floor
    plot_null_len = min(plot_horizon, len(winner_null_mean))
    x_axes = range(1, plot_null_len + 1)

    ax.plot(
        x_axes,
        winner_null_mean[:plot_null_len],
        color="#1A1A1A",
        linestyle="--",
        linewidth=2.4,
        label=f"Empirical Null ({winner_name[:18]}... N=100 Perms)",
    )
    ax.fill_between(
        x_axes,
        winner_null_mean[:plot_null_len] - (2 * winner_null_std[:plot_null_len]),
        winner_null_mean[:plot_null_len] + (2 * winner_null_std[:plot_null_len]),
        color="black",
        alpha=0.14,
        label=band_label,
    )

    # Contextual callouts
    if plot_null_len >= 3:
        noise_idx = max(1, int(plot_null_len * 0.7) - 1)
        ax.annotate(
            "Chance Alignment (p ≥ 0.05)",
            xy=(noise_idx + 1, winner_null_mean[noise_idx]),
            xytext=(noise_idx + 1, max(0.02, winner_null_mean[noise_idx] - 0.15)),
            ha="center",
            fontsize=9.5,
            fontstyle="italic",
            color="#444444",
            arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 1.0},
        )

        bio_idx = max(0, int(plot_null_len * 0.3) - 1)
        winner_y = decay_curves[winner_name][bio_idx]
        ax.annotate(
            "Verified Signal (p < 0.05)",
            xy=(bio_idx + 1, winner_y),
            xytext=(bio_idx + 1, min(0.95, winner_y + 0.15)),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#1D3557",
            arrowprops={"arrowstyle": "->", "color": "#1D3557", "lw": 1.2},
        )

    _add_dual_legends(
        ax,
        model_lines,
        models_title=models_legend_title,
        null_label=f"Empirical Null ({winner_name[:18]}...)",
        band_label=legend_band_label,
        controls_title="🎲 Statistical Control",
    )

    ax.set_title(title, fontsize=15, pad=14, fontweight="bold")
    ax.set_xlabel("Canonical Axis Rank", fontsize=12, labelpad=8)
    ax.set_ylabel("Canonical Correlation ($r$)", fontsize=12, labelpad=8)
    ax.set_xlim(1, max(plot_horizon, 2))
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.show()


def plot_svcca_decay_overview_v2(
    decay_curves: Mapping[str, np.ndarray], null_mean: np.ndarray, null_std: np.ndarray
) -> None:
    """SVCCA decay curves of all models against one null baseline (v2 figure layout).

    Unlike :func:`plot_svcca_decay_overview`, the callouts are placed at fixed positions
    (axes 25 and 35), so ``null_mean`` must cover at least 35 axes, and the x-axis always
    spans 50 axes.

    Parameters
    ----------
    decay_curves : mapping of str to numpy.ndarray
        Model name -> canonical correlations, strongest first.
    null_mean, null_std : numpy.ndarray
        Null baseline (mean and SD per rank) of the top-ranked model.
    """
    print("Generating SVCCA Axis Decay Plot with Enhanced Control Legends...")
    _, ax = plt.subplots(figsize=(14.5, 8), dpi=140)
    sns.set_style("whitegrid")

    model_lines = []
    for name, corrs in decay_curves.items():
        c, ls, lw, alpha = _model_style(name)
        (line,) = ax.plot(range(1, len(corrs) + 1), corrs, label=name, color=c, linestyle=ls, linewidth=lw, alpha=alpha)
        model_lines.append(line)

    # Empirical permutation baseline and noise floor
    x_axes = range(1, len(null_mean) + 1)
    ax.plot(
        x_axes,
        null_mean,
        color="#1A1A1A",
        linestyle="--",
        linewidth=2.4,
        label="Scrambled Pairing Mean (N=100 Permutations)",
    )
    ax.fill_between(
        x_axes,
        null_mean - (2 * null_std),
        null_mean + (2 * null_std),
        color="black",
        alpha=0.14,
        label="Noise Floor: 95% CI (Mean ± 2σ, p = 0.05 cutoff)",
    )

    # Contextual callout annotations
    ax.annotate(
        "Statistically Verified Biological Alignment (p < 0.05)",
        xy=(25, 0.58),
        xytext=(25, 0.70),
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#1D3557",
        arrowprops={"arrowstyle": "->", "color": "#1D3557", "lw": 1.2},
    )
    ax.annotate(
        "Chance Alignment / Mathematical Noise (p ≥ 0.05)",
        xy=(35, null_mean[34]),
        xytext=(35, null_mean[34] - 0.12),
        ha="center",
        fontsize=9.5,
        fontstyle="italic",
        color="#444444",
        arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 1.0},
    )

    _add_dual_legends(
        ax,
        model_lines,
        models_title="🧬 Morphological Representations",
        null_label="Scrambled Pairing Mean (N=100 Permutations)",
        band_label="Noise Floor: 95% CI (Mean ± 2σ, p = 0.05 cutoff)",
        controls_title="🎲 Background Control Experiment",
    )

    ax.set_title(
        "SVCCA Canonical Axis Decay: Morphology vs. DRUG-seq Transcriptomics\n"
        "(Pushdown Filtered & Median Replicate-Aggregated vs. Empirical Baseline)",
        fontsize=15,
        pad=14,
        fontweight="bold",
    )
    ax.set_xlabel("Canonical Axis Rank (Dimensions of Shared Variance)", fontsize=12, labelpad=8)
    ax.set_ylabel("Canonical Correlation ($r$)", fontsize=12, labelpad=8)
    ax.set_xlim(1, 50)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.show()


def plot_svcca_decay_grid(
    leaderboard_df: pd.DataFrame,
    decay_curves: Mapping[str, np.ndarray],
    null_curves: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """One panel per model: its SVCCA decay curve against its own permutation null.

    Panels follow the leaderboard order and are annotated with the model's adjusted AUC,
    number of significant axes and number of morphology principal components.

    Parameters
    ----------
    leaderboard_df : pandas.DataFrame
        Ranked results with a ``"Model / Processing"`` column and the SVCCA statistics.
    decay_curves : mapping of str to numpy.ndarray
        Model name -> canonical correlations, strongest first.
    null_curves : mapping of str to (numpy.ndarray, numpy.ndarray)
        Model name -> (null mean, null SD) per rank.
    """
    print("Generating Individual SVCCA Model Panels with Matched Noise Floors...")

    # Order models strictly by leaderboard rank
    sorted_models = leaderboard_df["Model / Processing"].tolist()
    n_models = len(sorted_models)

    # 2 Columns x N/2 Rows layout
    ncols = 2
    nrows = int(np.ceil(n_models / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4.2 * nrows), dpi=140, sharey=True)
    axes = axes.flatten()
    sns.set_style("whitegrid")

    for idx, name in enumerate(sorted_models):
        ax = axes[idx]

        # Extract model data and its matched empirical null
        true_corrs = decay_curves[name]
        null_mean, null_std = null_curves[name]

        # Retrieve stats from leaderboard
        row_stats = leaderboard_df[leaderboard_df["Model / Processing"] == name].iloc[0]
        adj_auc = float(row_stats.get("SVCCA Adj. AUC", 0.0))
        true_auc = float(row_stats.get("SVCCA True AUC", 0.0))
        null_auc = float(row_stats.get("SVCCA Null AUC", 0.0))
        sig_axes = int(row_stats.get("Significant Axes", 0))
        morph_pcs = int(row_stats.get("Morph PCs (90% Var)", row_stats.get("Morph PCs", 0)))

        horizon = min(len(true_corrs), len(null_mean))
        x_axis = np.arange(1, horizon + 1)

        # Architecture color
        c = next((col for k, col in MODEL_COLORS.items() if name.startswith(k)), "#333333")
        ls = ":" if "Raw" in name else "-"

        # 1. True SVCCA decay curve
        ax.plot(
            x_axis,
            true_corrs[:horizon],
            color=c,
            linestyle=ls,
            linewidth=2.2,
            label=f"True Signal (AUC: {true_auc:.3f})",
        )

        # 2. Model-specific matched null baseline
        ax.plot(
            x_axis,
            null_mean[:horizon],
            color="#222222",
            linestyle="--",
            linewidth=1.8,
            label=f"Matched Null Floor (AUC: {null_auc:.3f})",
        )

        # 3. Model-specific 95% noise floor (mean ± 2σ)
        ax.fill_between(
            x_axis,
            null_mean[:horizon] - (2 * null_std[:horizon]),
            null_mean[:horizon] + (2 * null_std[:horizon]),
            color="black",
            alpha=0.12,
            label="Noise Floor (±2σ, p = 0.05 cutoff)",
        )

        # 4. Vertical marker where signal meets noise
        if 0 < sig_axes < horizon:
            ax.axvline(
                x=sig_axes,
                color="gray",
                linestyle=":",
                linewidth=1.2,
                alpha=0.8,
                label=f"Cutoff ({sig_axes} Sig. Axes)",
            )

        # Title and metric badges
        rank_badge = f"Rank {idx + 1}: {name}"
        stat_badge = f"Adj. AUC: {adj_auc:+.4f} | Sig Axes: {sig_axes} | Morph PCs: {morph_pcs}"
        ax.set_title(f"{rank_badge}\n{stat_badge}", fontsize=11.5, fontweight="bold", pad=8)

        ax.set_xlim(1, horizon)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Canonical Axis Rank", fontsize=10, labelpad=4)
        ax.set_ylabel("Canonical Correlation ($r$)", fontsize=10, labelpad=4)
        ax.legend(loc="upper right", fontsize=8.5, frameon=True, framealpha=0.9)

    # Clean up empty subplots if number of models is odd
    for j in range(n_models, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        "Individual SVCCA Model Decays vs. Model-Specific Permutation Baselines\n"
        "(Ordered from Highest to Lowest Adjusted Biological Signal)",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()
    plt.show()


def plot_svcca_floor_grid(
    leaderboard_df: pd.DataFrame,
    decay_curves: Mapping[str, np.ndarray],
    null_curves: Mapping[str, tuple[np.ndarray, np.ndarray]],
    cutoff_curves: Mapping[str, np.ndarray],
    floor_label: str,
    subtitle: str,
    max_axes: int = 50,
    ncols: int = 4,
) -> None:
    """One panel per model: its decay curve, its null mean and a configurable noise floor.

    Unlike :func:`plot_svcca_decay_grid`, the shaded band runs from the null mean up to the
    significance cut-off of the benchmark (e.g. null mean + 3 SD, or the 95th percentile), and
    panels follow the order of ``decay_curves``.

    Parameters
    ----------
    leaderboard_df : pandas.DataFrame
        Results with ``"Model / Processing"``, ``"SVCCA Adj. AUC"``, ``"Significant Axes"``,
        ``"Compounds (N)"`` and ``"Evaluated Dims"`` columns.
    decay_curves : mapping of str to numpy.ndarray
        Model name -> canonical correlations, strongest first.
    null_curves : mapping of str to (numpy.ndarray, numpy.ndarray)
        Model name -> (null mean, null SD) per rank.
    cutoff_curves : mapping of str to numpy.ndarray
        Model name -> significance cut-off per rank.
    floor_label : str
        Legend text of the noise-floor line, e.g. ``"Noise Floor (+3σ)"``.
    subtitle : str
        Second line of the figure title (the benchmark settings).
    max_axes : int, default 50
        Number of canonical axes shown at most.
    ncols : int, default 4
        Panels per row.
    """
    n_models = len(decay_curves)
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows), dpi=140, sharey=True, squeeze=False)
    axes = axes.flatten()
    plot_horizon = min(max_axes, max(len(corrs) for corrs in decay_curves.values()))

    for idx, (name, corrs) in enumerate(decay_curves.items()):
        ax = axes[idx]
        null_mean, _ = null_curves[name]
        cutoff = cutoff_curves[name]
        row = leaderboard_df[leaderboard_df["Model / Processing"] == name].iloc[0]
        sig_axes = int(row["Significant Axes"])

        c = next((col for k, col in MODEL_COLORS.items() if name.startswith(k)), "#2B2D42")
        ls = ":" if "Raw" in name else "-"
        plot_len = min(plot_horizon, len(corrs))
        x_range = np.arange(1, plot_len + 1)

        ax.plot(x_range, corrs[:plot_len], label="True Biological Decay", color=c, linestyle=ls, linewidth=2.4)
        ax.plot(
            x_range,
            null_mean[:plot_len],
            color="#2B2D42",
            linestyle="--",
            linewidth=1.6,
            label="Empirical Null (Mean)",
        )
        ax.plot(x_range, cutoff[:plot_len], color="black", linestyle=":", linewidth=1.2, label=floor_label)
        ax.fill_between(x_range, null_mean[:plot_len], cutoff[:plot_len], color="black", alpha=0.12)

        if 0 < sig_axes <= plot_len:
            ax.axvline(x=sig_axes, color=c, linestyle="-.", alpha=0.7, linewidth=1.5, label=f"Cutoff ({sig_axes} Axes)")
            ax.scatter([sig_axes], [corrs[sig_axes - 1]], color=c, s=40, zorder=5)

        ax.set_title(
            f"{name}\nN={int(row['Compounds (N)'])} | D={int(row['Evaluated Dims'])} | "
            f"Adj AUC: {row['SVCCA Adj. AUC']:+.4f} | Sig: {sig_axes}",
            fontsize=10.5,
            fontweight="bold",
            pad=8,
        )
        ax.set_xlabel("Axis Rank", fontsize=10)
        if idx % ncols == 0:
            ax.set_ylabel("Canonical Correlation ($r$)", fontsize=10)
        ax.set_xlim(1, plot_len)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=7.5, frameon=True)

    for j in range(n_models, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        f"SVCCA Canonical Axis Decay vs. Empirical Permutation Floors\n{subtitle}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.show()
