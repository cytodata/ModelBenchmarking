"""Test whether a canonical axis tracks a known gene module or a known drug class.

Axis decoding (:mod:`model_benchmarking.decoding`) starts from the data and asks what an axis
means. The functions here go the other way and start from a biological hypothesis, here
microtubule/mitotic-arrest biology:

* **Gene module.** Score every compound by the average (z-scored) expression of a gene set and
  find the axes that correlate with that score.
* **Reference compounds.** Mark compounds of a known class (e.g. tubulin binders) and find the axes
  on which they score unusually high or low (ROC-AUC, effect size and rank tests).

An axis supporting both would be a convincing readout of that biology. Axes have an arbitrary
sign, so every scan reports the polarity that separates the hypothesis best.
"""

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp, mannwhitneyu, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from model_benchmarking.annotation import get_rich_compound_annotation

# Spindle assembly checkpoint, mitotic kinases and cyclins, kinesins, kinetochore/centrosome, tubulins
# and microtubule regulators, plus some older gene aliases.
MICROTUBULE_GENES = [
    "BUB1", "BUB1B", "BUB3", "MAD2L1", "MAD1L1", "CDC20", "TTK", "CENPE", "ZWILCH", "ZWINT",
    "AURKA", "AURKB", "PLK1", "PLK4", "CDK1", "CCNB1", "CCNB2", "CCNA2", "CDC25C",
    "KIF11", "KIF2C", "KIF4A", "KIF15", "KIF18A", "KIF20A", "KIF22", "KIF23", "KIFC1",
    "NDC80", "NUF2", "SPC24", "SPC25", "CENPA", "SKA1", "SKA2", "SKA3", "TPX2", "PRC1",
    "TUBA1A", "TUBA1B", "TUBA1C", "TUBB", "TUBB2A", "TUBB2B", "TUBB3", "TUBB4B", "TUBG1",
    "MAP4", "STMN1", "CKAP5", "EB1", "MAPRE1",
    "EG5", "CDC2", "MPS1", "STK15", "BIRC5",
]  # fmt: skip

# Microtubule-targeting drugs and spindle (Eg5/KIF11) inhibitors, keyed by InChIKey. Keys were
# looked up by drug name in PubChem (2026-10-01); matching also falls back to the first 14
# characters (the molecular skeleton), so stereoisomers and salts of the same drug are found.
TUBULIN_DRUGS: dict[str, dict[str, str]] = {
    "RCINICONZNJXQF-MZXODVADSA-N": {"name": "Paclitaxel", "class": "Taxane (Stabilizer)"},
    "ZDZOTLJHXYCWBA-VCVYQWHSSA-N": {"name": "Docetaxel", "class": "Taxane (Stabilizer)"},
    "BMQGVNUXMIRLCK-OAGWZNDDSA-N": {"name": "Cabazitaxel", "class": "Taxane (Stabilizer)"},
    "JXLYSJRDGCGARV-CFWMRBGOSA-N": {"name": "Vinblastine", "class": "Vinca Alkaloid (Depolymerizer)"},
    "OGWKCGZFUXNPDA-XQKSVPLYSA-N": {"name": "Vincristine", "class": "Vinca Alkaloid (Depolymerizer)"},
    "GBABOYUKABKIAF-IELIFDKJSA-N": {"name": "Vinorelbine", "class": "Vinca Alkaloid (Depolymerizer)"},
    "HHJUWIANJFBDHT-KOTLKJBCSA-N": {"name": "Vindesine", "class": "Vinca Alkaloid (Depolymerizer)"},
    "IAKHMKGGTNLKSZ-INIZCTEOSA-N": {"name": "Colchicine", "class": "Colchicine Site"},
    "KYRVNWMVYQXFEU-UHFFFAOYSA-N": {"name": "Nocodazole", "class": "Synthetic Depolymerizer"},
    "YJGVMLPVUAXIQN-XVVDYKMHSA-N": {"name": "Podophyllotoxin", "class": "Colchicine Site"},
    "HVXBOLULGPECHP-WAYWQWQTSA-N": {"name": "Combretastatin A-4", "class": "Colchicine Site"},
    "URCVCIZFVQDVPM-UHFFFAOYSA-N": {"name": "ABT-751 (E7010)", "class": "Colchicine Site"},
    "QXRSDHAAWVKZLJ-PVYNADRNSA-N": {"name": "Epothilone B (Patupilone)", "class": "Epothilone (Stabilizer)"},
    "FABUFPQFXZVHFB-PVYNADRNSA-N": {"name": "Ixabepilone", "class": "Epothilone (Stabilizer)"},
    "UFNVPOGXISZXJD-JBQZKEIOSA-N": {"name": "Eribulin", "class": "Halichondrin (Depolymerizer)"},
    "HDDSHPAODJUKPD-UHFFFAOYSA-N": {"name": "Fenbendazole", "class": "Benzimidazole Tubulin-i"},
    "HXHWSAZORRCQMX-UHFFFAOYSA-N": {"name": "Albendazole", "class": "Benzimidazole Tubulin-i"},
    "OPXLLQIJSORQAM-UHFFFAOYSA-N": {"name": "Mebendazole", "class": "Benzimidazole Tubulin-i"},
    "DDUHZTYCFQRHIY-RBHXEPJQSA-N": {"name": "Griseofulvin", "class": "Tubulin Assembly-i"},
    "LOBCDGHHHHGHFA-LBPRGKRZSA-N": {"name": "Monastrol", "class": "KIF11/Eg5 Spindle Inhibitor"},
    "QJZRFPJCWMNVAV-HHHXNRCGSA-N": {"name": "Ispinesib", "class": "KIF11/Eg5 Spindle Inhibitor"},
    "LLXISKGBWFTGEI-FQEVSTJZSA-N": {"name": "Filanesib", "class": "KIF11/Eg5 Spindle Inhibitor"},
}


def resolve_gene_columns(
    genes: Sequence[str], gene_columns: Sequence[str], ensembl_to_symbol: Mapping[str, str]
) -> dict[str, str]:
    """Find the table columns holding the genes of a signature.

    Parameters
    ----------
    genes : sequence of str
        Gene symbols of the signature (case-insensitive).
    gene_columns : sequence of str
        Column names of the gene matrix: Ensembl IDs (version suffix allowed) or already symbols.
    ensembl_to_symbol : mapping of str to str
        Ensembl ID (without version) -> gene symbol.

    Returns
    -------
    dict of str to str
        Upper-case gene symbol -> column name, for the genes that are present.
    """
    symbol_to_column = {}
    for col in gene_columns:
        symbol = ensembl_to_symbol.get(col.split(".")[0].strip())
        if symbol:
            symbol_to_column[symbol.upper().strip()] = col
        symbol_to_column[col.upper().strip()] = col

    matched = {}
    for gene in genes:
        key = gene.upper().strip()
        if key in symbol_to_column:
            matched[key] = symbol_to_column[key]
    return matched


def module_scores(expression: np.ndarray) -> np.ndarray:
    """Per-compound activation of a gene module: the mean across genes of the z-scored expression.

    Parameters
    ----------
    expression : numpy.ndarray, shape (n_compounds, n_module_genes)
        Missing values are treated as 0 and constant genes are not rescaled.

    Returns
    -------
    numpy.ndarray, shape (n_compounds,)
    """
    values = np.nan_to_num(np.asarray(expression, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    stds = values.std(axis=0)
    stds[stds < 1e-8] = 1.0
    return np.nan_to_num((values - values.mean(axis=0)) / stds).mean(axis=1)


def scan_axes_for_module(
    axis_scores: pd.DataFrame, module_score: np.ndarray, canonical_r: Sequence[float] | None = None
) -> pd.DataFrame:
    """Correlate every axis with a gene-module score.

    Parameters
    ----------
    axis_scores : pandas.DataFrame
        One column per axis, one row per compound (same order as ``module_score``).
    module_score : numpy.ndarray, shape (n_compounds,)
    canonical_r : sequence of float, optional
        Canonical correlation of each axis, in column order (meaningless after unmixing).

    Returns
    -------
    pandas.DataFrame
        One row per axis, sorted by ``|Pearson r|``: ``Axis``, ``Axis_Index``, ``Canonical_r``,
        ``Module_Corr_Pearson``, ``Module_Corr_Spearman``, ``T_Statistic``, ``P_Value``, ``Abs_Pearson``.
    """
    n_samples = len(module_score)
    records = []
    for idx, col in enumerate(axis_scores.columns):
        x = np.nan_to_num(axis_scores[col].values.astype(np.float64))
        finite = np.isfinite(x) & np.isfinite(module_score)
        x, y = x[finite], module_score[finite]
        if np.std(x) > 1e-8 and np.std(y) > 1e-8:
            r_pearson, p_value = pearsonr(x, y)
            r_spearman, _ = spearmanr(x, y)
        else:
            r_pearson, p_value, r_spearman = 0.0, 1.0, 0.0
        records.append(
            {
                "Axis": col,
                "Axis_Index": idx,
                "Canonical_r": canonical_r[idx] if canonical_r is not None and idx < len(canonical_r) else np.nan,
                "Module_Corr_Pearson": r_pearson,
                "Module_Corr_Spearman": r_spearman,
                "T_Statistic": r_pearson * np.sqrt((n_samples - 2) / (1.0 - np.clip(r_pearson**2, 0, 0.9999))),
                "P_Value": p_value,
            }
        )
    scan = pd.DataFrame(records)
    scan["Abs_Pearson"] = scan["Module_Corr_Pearson"].abs()
    return scan.sort_values(by="Abs_Pearson", ascending=False).reset_index(drop=True)


def print_module_drivers(
    table: pd.DataFrame, axis: str, polarity: float, module_score: np.ndarray, merge_key: str, top_n: int = 10
) -> None:
    """Print the compounds scoring highest on an axis (in its module-supporting direction), annotated.

    Parameters
    ----------
    table : pandas.DataFrame
        Compounds with ``merge_key`` and the axis column, same order as ``module_score``.
    axis : str
        Axis column.
    polarity : float
        +1 or -1: sign applied to the axis so that high scores mean high module activity.
    module_score : numpy.ndarray
    merge_key : str
    top_n : int, default 10
    """
    frame = pd.DataFrame(
        {merge_key: table[merge_key].values, "axis": table[axis].values * polarity, "module": module_score}
    )
    for rank, (_, row) in enumerate(frame.sort_values("axis", ascending=False).head(top_n).iterrows(), 1):
        name, moa = get_rich_compound_annotation(row[merge_key])
        print(f"  {rank:2d}. {name:<24} | {moa:<38} | Axis: {row['axis']:+.3f} | Module: {row['module']:+.2f}")


def match_reference_compounds(
    inchikeys: Sequence[str], reference: Mapping[str, Mapping[str, str]]
) -> tuple[np.ndarray, list[tuple[str, str, str] | None]]:
    """Find which compounds belong to a reference set, by exact InChIKey and then by first block.

    Parameters
    ----------
    inchikeys : sequence of str
        Compounds of the screen.
    reference : mapping of str to mapping
        InChIKey -> ``{"name": ..., "class": ...}``.

    Returns
    -------
    is_reference : numpy.ndarray of bool
    details : list
        For each compound ``(name, class, "Exact 27-char" | "Block-1 Rescue")`` or ``None``.
    """
    block1 = {k[:14]: v for k, v in reference.items()}
    flags, details = [], []
    for ikey in inchikeys:
        key = str(ikey).strip()
        if key in reference:
            meta = reference[key]
            flags.append(True)
            details.append((meta["name"], meta["class"], "Exact 27-char"))
        elif key[:14] in block1:
            meta = block1[key[:14]]
            flags.append(True)
            details.append((meta["name"], meta["class"], "Block-1 Rescue"))
        else:
            flags.append(False)
            details.append(None)
    return np.array(flags), details


def scan_axes_for_reference_set(
    axis_scores: pd.DataFrame, is_reference: np.ndarray, canonical_r: Sequence[float] | None = None
) -> pd.DataFrame:
    """Test every axis for a shift in the scores of a reference set of compounds.

    For each axis: ROC-AUC (probability that a random reference compound outscores a random other
    compound), Cohen's d, a Mann-Whitney test and the Kolmogorov-Smirnov statistic. Axes have an
    arbitrary sign, so ROC-AUC is reported for the better of the two directions (``Polarity``
    tells which) together with the matching one-sided p-value.

    Parameters
    ----------
    axis_scores : pandas.DataFrame
        One column per axis, one row per compound.
    is_reference : numpy.ndarray of bool
        Compounds belonging to the reference set (at least 2).
    canonical_r : sequence of float, optional
        Canonical correlation of each axis, in column order.

    Returns
    -------
    pandas.DataFrame
        One row per axis, sorted by ROC-AUC.
    """
    n_total = len(axis_scores)
    records = []
    for idx, col in enumerate(axis_scores.columns):
        scores = np.nan_to_num(axis_scores[col].values.astype(np.float64))
        gt, bg = scores[is_reference], scores[~is_reference]

        ks_stat, _ = ks_2samp(gt, bg)
        pooled_std = np.sqrt(((len(gt) - 1) * np.var(gt) + (len(bg) - 1) * np.var(bg)) / (n_total - 2))
        cohen_d = (np.mean(gt) - np.mean(bg)) / (pooled_std + 1e-12)

        roc_auc = roc_auc_score(is_reference.astype(int), scores)
        if roc_auc < 0.5:
            effective_auc, polarity = 1.0 - roc_auc, -1.0
            _, best_p = mannwhitneyu(gt, bg, alternative="less")
        else:
            effective_auc, polarity = roc_auc, 1.0
            _, best_p = mannwhitneyu(gt, bg, alternative="greater")

        records.append(
            {
                "Axis": col,
                "Axis_Index": idx,
                "Canonical_r": canonical_r[idx] if canonical_r is not None and idx < len(canonical_r) else np.nan,
                "Cohen_d": cohen_d,
                "Abs_Cohen_d": abs(cohen_d),
                "Polarity": "Positive (+)" if polarity > 0 else "Negative (-)",
                "ROC_AUC": effective_auc,
                "P_Value": best_p,
                "KS_Stat": ks_stat,
                "GT_Mean": np.mean(gt),
                "BG_Mean": np.mean(bg),
            }
        )
    return pd.DataFrame(records).sort_values(by="ROC_AUC", ascending=False).reset_index(drop=True)


def reference_module_separation(
    module_score: np.ndarray, is_reference: np.ndarray
) -> tuple[float, float, float] | None:
    """Test whether reference compounds score higher on a gene module than other compounds.

    Returns
    -------
    reference_mean, background_mean : float
        Mean module score of the two groups.
    p_value : float
        One-sided Mann-Whitney p-value (reference higher).
    Returns ``None`` if fewer than 2 reference compounds are available.
    """
    if np.sum(is_reference) < 2:
        return None
    module = np.nan_to_num(module_score)
    _, p_value = mannwhitneyu(module[is_reference], module[~is_reference], alternative="greater")
    return float(np.mean(module[is_reference])), float(np.mean(module[~is_reference])), float(p_value)


def plot_module_scan(
    merged: pd.DataFrame, module_score: np.ndarray, scan: pd.DataFrame, title: str = "Microtubule Signature"
) -> None:
    """Scatter of the best axis against the module score, and |r| of every axis with the module.

    Parameters
    ----------
    merged : pandas.DataFrame
        Compounds with one column per axis, same order as ``module_score``.
    module_score : numpy.ndarray
    scan : pandas.DataFrame
        Output of :func:`scan_axes_for_module`.
    title : str
    """
    best = scan.iloc[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), dpi=120)

    sns.regplot(
        x=merged[best["Axis"]],
        y=module_score,
        ax=ax1,
        color="#1D3557",
        scatter_kws={"alpha": 0.35, "s": 20},
        line_kws={"color": "#E63946", "linewidth": 2},
    )
    ax1.set_title(
        f"{best['Axis']} vs. {title}\n(Pearson r = {best['Module_Corr_Pearson']:+.3f}, p = {best['P_Value']:.2e})",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_xlabel(f"{best['Axis']} Score", fontsize=10)
    ax1.set_ylabel("Module Z-Score", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    by_index = scan.sort_values("Axis_Index")
    colors = ["#E63946" if a == best["Axis"] else "#2B2D42" for a in by_index["Axis"]]
    ax2.bar(range(1, len(by_index) + 1), by_index["Abs_Pearson"], color=colors, alpha=0.85)
    ax2.set_title(f"{title} Coupling Across All Axes", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Axis Rank (1 to K)", fontsize=10)
    ax2.set_ylabel("|Pearson Correlation| with Module", fontsize=10)
    ax2.set_xticks(range(1, len(by_index) + 1))
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_reference_scan(
    axis_scores: pd.DataFrame, is_reference: np.ndarray, scan: pd.DataFrame, label: str = "Microtubule Inhibitors"
) -> None:
    """ROC-AUC of every axis, and the score distribution of the best axis for reference vs. other compounds.

    Parameters
    ----------
    axis_scores : pandas.DataFrame
        One column per axis, one row per compound.
    is_reference : numpy.ndarray of bool
    scan : pandas.DataFrame
        Output of :func:`scan_axes_for_reference_set`.
    label : str
        Name of the reference set.
    """
    best = scan.iloc[0]
    polarity = 1.0 if "Positive" in best["Polarity"] else -1.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), dpi=120)

    by_index = scan.sort_values("Axis_Index")
    colors = ["#E63946" if a == best["Axis"] else "#2B2D42" for a in by_index["Axis"]]
    ax1.bar(range(1, len(by_index) + 1), by_index["ROC_AUC"], color=colors, alpha=0.85)
    ax1.axhline(y=0.50, color="gray", linestyle="--", linewidth=1, label="Random Chance (AUC = 0.50)")
    ax1.set_title(
        f"{label} Retrieval Capacity (ROC-AUC)\nPrimary Dimension: {best['Axis']} (AUC = {best['ROC_AUC']:.3f})",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_xlabel("Axis Index (1 to K)", fontsize=10)
    ax1.set_ylabel("ROC-AUC Score", fontsize=10)
    ax1.set_ylim(0.40, 1.02)
    ax1.set_xticks(range(1, len(by_index) + 1))
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.4)

    oriented = axis_scores[best["Axis"]].values * polarity
    sns.kdeplot(
        oriented[~is_reference],
        ax=ax2,
        label="Background Compounds",
        color="#2B2D42",
        fill=True,
        alpha=0.2,
        linewidth=1.5,
    )
    sns.kdeplot(oriented[is_reference], ax=ax2, label=label, color="#E63946", fill=True, alpha=0.5, linewidth=2.4)
    ax2.axvline(
        x=np.mean(oriented[is_reference]), color="#E63946", linestyle="--", linewidth=1.4, label="Reference Mean"
    )
    ax2.axvline(
        x=np.mean(oriented[~is_reference]), color="#2B2D42", linestyle=":", linewidth=1.2, label="Background Mean"
    )
    ax2.set_title(
        f"Separation along Oriented {best['Axis']}\n(Cohen's d: {best['Cohen_d']:+.2f} | p = {best['P_Value']:.2e})",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_xlabel(f"Oriented {best['Axis']} Score", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()
