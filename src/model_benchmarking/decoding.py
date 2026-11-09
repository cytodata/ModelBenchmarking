"""Interpret shared morphology-transcriptomics axes ("axis decoding").

For one canonical axis (a direction along which image embeddings and gene expression
co-vary), these functions report:

* which genes follow the axis (Pearson correlation and t-statistic per gene),
* which pathways are enriched among those genes (GSEA pre-ranked, Enrichr ORA),
* whether the compounds scoring highest on the axis ("drivers") resemble each other in
  a reference space such as Chemical Checker bioactivity or MiniMol structure,
* what those driver compounds are (PubChem / ChEMBL annotation),
* whether the drivers also look alike in morphology (phenotypic coherence),
* a volcano plot of the gene statistics.

GSEA and ORA call the Enrichr web service through ``gseapy``, and annotation calls
PubChem/ChEMBL, so an internet connection is required.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import mannwhitneyu
from scipy.stats import t as t_dist

from model_benchmarking.annotation import get_rich_compound_annotation
from model_benchmarking.chemistry import chemistry_verdict, driver_tanimoto
from model_benchmarking.neighborhoods import get_pairwise_sims
from model_benchmarking.utils import get_device


@dataclass(frozen=True)
class DriverReportStyle:
    """Printed labels for :func:`report_driver_similarity` in one reference space."""

    title: str
    similarity_label: str
    verdict_distinct: str
    verdict_weak: str
    verdict_diverse: str


CHEMICAL_CHECKER_STYLE = DriverReportStyle(
    title="🧪 CHEMICAL CHECKER SIGNATURE:",
    similarity_label="Mean CC Cosine Similarity  ",
    verdict_distinct="✅ DISTINCT TARGET / NETWORK CLUSTER (Shared Mechanism)",
    verdict_weak="⚠️ WEAK CLUSTERING (Shared target subset / Broad network)",
    verdict_diverse="💡 DIVERSE CHEMISTRY (Scaffold Hopping / Phenotypic Convergence)",
)

MINIMOL_STYLE = DriverReportStyle(
    title="🧪 MINIMOL 2D STRUCTURAL SIGNATURE:",
    similarity_label="Mean Minimol Cosine Sim    ",
    verdict_distinct="✅ DISTINCT 2D PHARMACOPHORE CLUSTER (Rigid Chemical Scaffold)",
    verdict_weak="⚠️ WEAK STRUCTURAL CLUSTERING (Shared Sub-structure / Heterogeneous Series)",
    verdict_diverse="💡 DIVERSE 2D CHEMISTRY (Scaffold Hopping / Phenotypic Convergence)",
)


def gene_axis_statistics(
    axis_values: np.ndarray,
    Y_c_genes: torch.Tensor,
    var_Y_genes: torch.Tensor,
    gene_ids: Sequence[str],
    ensembl_to_symbol: Mapping[str, str],
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Correlate every gene with a canonical axis.

    Parameters
    ----------
    axis_values : numpy.ndarray, shape (n_compounds,)
        Compound scores on the axis.
    Y_c_genes : torch.Tensor, shape (n_compounds, n_genes)
        Column-centered gene-expression matrix (same compound order).
    var_Y_genes : torch.Tensor, shape (n_genes,)
        Sum of squares of each centered gene column.
    gene_ids : sequence of str
        Ensembl IDs of the gene columns (version suffixes allowed).
    ensembl_to_symbol : mapping of str to str
        Ensembl ID (without version) -> gene symbol; unmapped genes keep their ID.
    device : torch.device, optional
        Defaults to :func:`model_benchmarking.utils.get_device`.

    Returns
    -------
    pandas.DataFrame
        Columns ``Ensembl_ID``, ``Gene_Symbol``, ``Correlation``, ``T_Stat``,
        ``P_Value`` (two-sided) and ``Neg_Log10_P``.
    """
    device = get_device() if device is None else device
    axis_tensor = torch.tensor(np.asarray(axis_values).astype(np.float64), device=device)
    X_c_axis = axis_tensor - axis_tensor.mean()
    var_X_axis = torch.sum(X_c_axis**2)

    cov = torch.matmul(X_c_axis, Y_c_genes)
    r_tensor = cov / torch.sqrt(var_X_axis * var_Y_genes + 1e-12)

    n_samples = len(axis_values)
    r_clipped = torch.clamp(r_tensor, -0.9999, 0.9999)

    # T-statistic of the correlation (used as a ranking metric)
    t_stat = r_clipped * torch.sqrt((n_samples - 2) / (1.0 - r_clipped**2))

    r_vals = r_tensor.cpu().numpy()
    t_vals = t_stat.cpu().numpy()
    p_vals = 2 * t_dist.sf(np.abs(t_vals), df=n_samples - 2)
    neg_log10_p = -np.log10(p_vals + 1e-300)

    symbols = [ensembl_to_symbol.get(g.split(".")[0], g) for g in gene_ids]
    return pd.DataFrame(
        {
            "Ensembl_ID": gene_ids,
            "Gene_Symbol": symbols,
            "Correlation": r_vals,
            "T_Stat": t_vals,
            "P_Value": p_vals,
            "Neg_Log10_P": neg_log10_p,
        }
    )


def rank_genes(gene_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Gene ranking for GSEA: one row per gene symbol, sorted by ``metric`` (high to low).

    Genes without a symbol (still named ``ENSG...``) or with a missing correlation are
    dropped; when several Ensembl IDs share a symbol, the one with the largest
    ``|metric|`` is kept.

    Parameters
    ----------
    gene_df : pandas.DataFrame
        Output of :func:`gene_axis_statistics`.
    metric : {"T_Stat", "Correlation"}

    Returns
    -------
    pandas.DataFrame
        Columns ``Gene_Symbol`` and ``metric``.
    """
    valid_genes = gene_df.dropna(subset=["Correlation"]).copy()
    valid_genes = valid_genes[~valid_genes["Gene_Symbol"].str.startswith("ENSG")].copy()
    valid_genes["abs_metric"] = valid_genes[metric].abs()

    dedup_ranks = valid_genes.sort_values(
        by=["abs_metric", "Gene_Symbol"], ascending=[False, True], kind="stable"
    ).drop_duplicates("Gene_Symbol")
    return dedup_ranks[["Gene_Symbol", metric]].sort_values(
        by=[metric, "Gene_Symbol"], ascending=[False, True], kind="stable"
    )


def print_top_genes(top_pos_genes: pd.DataFrame, top_neg_genes: pd.DataFrame) -> None:
    """Print the most positively and negatively correlated genes of an axis."""
    print("🧬 TRANSCRIPTIONAL SIGNATURE:")
    print(
        "  ↳ Key Upregulated   : "
        + ", ".join([f"{r.Gene_Symbol} ({r.Correlation:+.2f})" for _, r in top_pos_genes.iterrows()])
    )
    print(
        "  ↳ Key Downregulated : "
        + ", ".join([f"{r.Gene_Symbol} ({r.Correlation:+.2f})" for _, r in top_neg_genes.iterrows()])
    )


def print_gsea(
    rnk_input: pd.DataFrame,
    gene_sets: str,
    positive_header: str,
    negative_header: str,
    term_width: int,
) -> None:
    """Run GSEA pre-ranked and print the 5 most positively and negatively enriched gene sets.

    Parameters
    ----------
    rnk_input : pandas.DataFrame
        Gene ranking from :func:`rank_genes`.
    gene_sets : str
        Enrichr library name, e.g. ``"MSigDB_Hallmark_2020"``.
    positive_header, negative_header : str
        Section titles printed above each list.
    term_width : int
        Width at which gene-set names are truncated and padded.

    Notes
    -----
    Uses 200 permutations, gene sets of 10-500 genes and seed 42. Errors (e.g. no
    network) are printed instead of raised.
    """
    try:
        pre_res = gp.prerank(
            rnk=rnk_input,
            gene_sets=gene_sets,
            threads=1,
            min_size=10,
            max_size=500,
            permutation_num=200,
            outdir=None,
            seed=42,
        )
        res_df = pre_res.res2d
        top_pos_pw = res_df[res_df["NES"] > 0].sort_values(by="NES", ascending=False, kind="stable").head(5)
        top_neg_pw = res_df[res_df["NES"] < 0].sort_values(by="NES", ascending=True, kind="stable").head(5)

        for header, pathways in [(positive_header, top_pos_pw), (negative_header, top_neg_pw)]:
            print(f"\n  {header}")
            if len(pathways) > 0:
                for _, r_pw in pathways.iterrows():
                    term_clean = r_pw["Term"].replace("HALLMARK_", "")
                    print(
                        f"     • {term_clean[:term_width]:<{term_width}} "
                        f"| NES: {r_pw['NES']:+.2f} | FDR: {r_pw['FDR q-val']:.2e}"
                    )
            else:
                print("     • (None detected)")
    except Exception as e:
        print(f"  ↳ GSEA note: {e}")


def print_ora(rnk_input: pd.DataFrame, gene_sets: str = "MSigDB_Hallmark_2020") -> None:
    """Over-representation analysis (Enrichr) of the 100 top- and bottom-ranked genes.

    Prints the two most significant gene sets for each end of the ranking. Errors are
    printed instead of raised.
    """
    try:
        top_100_up = rnk_input.head(100)["Gene_Symbol"].tolist()
        top_100_down = rnk_input.tail(100)["Gene_Symbol"].tolist()

        enr_up = gp.enrichr(gene_list=top_100_up, gene_sets=gene_sets, outdir=None)
        enr_down = gp.enrichr(gene_list=top_100_down, gene_sets=gene_sets, outdir=None)

        up_hits = enr_up.res2d.sort_values("Adjusted P-value").head(2)
        down_hits = enr_down.res2d.sort_values("Adjusted P-value").head(2)

        print("\n  🔬 Direct Outlier ORA (Top 100 Genes via Fisher's Exact):")
        if len(up_hits) > 0:
            up_str = ", ".join(
                [f"{r.Term.replace('HALLMARK_', '')} (p={r['Adjusted P-value']:.1e})" for _, r in up_hits.iterrows()]
            )
            print(f"     ↳ Upregulated Program   : {up_str}")
        if len(down_hits) > 0:
            down_str = ", ".join(
                [f"{r.Term.replace('HALLMARK_', '')} (p={r['Adjusted P-value']:.1e})" for _, r in down_hits.iterrows()]
            )
            print(f"     ↳ Downregulated Program : {down_str}")
    except Exception as e:
        print(f"  ↳ ORA note: {e}")


def select_driver_compounds(
    bio_scores_df: pd.DataFrame, axis_name: str, reference_df: pd.DataFrame, merge_key: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compounds scoring highest on an axis ("drivers"), joined to a reference space.

    Drivers are the compounds scoring above mean + 2 SD on the axis (the top 15 if fewer than 5
    qualify).

    Parameters
    ----------
    bio_scores_df : pandas.DataFrame
        Compound scores on each axis plus ``merge_key``.
    axis_name : str
        Column of ``bio_scores_df`` to analyze.
    reference_df : pandas.DataFrame
        Reference embeddings with ``merge_key``.
    merge_key : str
        Compound identifier column.

    Returns
    -------
    merged : pandas.DataFrame
        All compounds present in both tables, one row per compound, highest score first.
    top_drivers : pandas.DataFrame
        The driver compounds (a subset of ``merged``).
    """
    merged = pd.merge(bio_scores_df[[merge_key, axis_name]], reference_df, on=merge_key, how="inner")
    merged = merged.sort_values(by=[axis_name, merge_key], ascending=[False, True], kind="stable").drop_duplicates(
        subset=[merge_key], keep="first"
    )

    scores = merged[axis_name].values
    cutoff = np.mean(scores) + (2.0 * np.std(scores))
    top_drivers = merged[merged[axis_name] > cutoff].copy()
    if len(top_drivers) < 5:
        top_drivers = merged.head(15).copy()
    return merged, top_drivers


def report_driver_similarity(
    bio_scores_df: pd.DataFrame,
    axis_name: str,
    reference_df: pd.DataFrame,
    reference_features: Sequence[str],
    merge_key: str,
    style: DriverReportStyle,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Test whether an axis's driver compounds are similar in a reference space.

    Drivers are the compounds scoring above mean + 2 SD on the axis (the top 15 if fewer
    than 5 qualify). Their pairwise cosine similarities in the reference space are
    compared with those of an equally sized random sample (one-sided Mann-Whitney test),
    and a verdict is printed.

    Parameters
    ----------
    bio_scores_df : pandas.DataFrame
        Compound scores on each axis plus ``merge_key``.
    axis_name : str
        Column of ``bio_scores_df`` to analyze.
    reference_df : pandas.DataFrame
        Reference embeddings (e.g. Chemical Checker) with ``merge_key``.
    reference_features : sequence of str
        Embedding columns of ``reference_df``.
    merge_key : str
        Compound identifier column.
    style : DriverReportStyle
        Printed labels (:data:`CHEMICAL_CHECKER_STYLE` or :data:`MINIMOL_STYLE`).
    device : torch.device, optional
        Defaults to :func:`model_benchmarking.utils.get_device`.

    Returns
    -------
    pandas.DataFrame
        The driver compounds, highest score first.

    Notes
    -----
    Pairwise similarities are not independent observations, so the p-value is
    optimistic; only the positive end of the axis is examined.
    """
    device = get_device() if device is None else device
    merged, top_drivers = select_driver_compounds(bio_scores_df, axis_name, reference_df, merge_key)

    k_drivers = len(top_drivers)

    drivers_tensor = torch.tensor(top_drivers[reference_features].values.astype(np.float32), device=device)
    random_group = merged.sample(n=k_drivers, random_state=42)
    random_tensor = torch.tensor(random_group[reference_features].values.astype(np.float32), device=device)

    driver_sims = get_pairwise_sims(drivers_tensor)
    random_sims = get_pairwise_sims(random_tensor)

    mean_driver_sim = np.mean(driver_sims)
    mean_random_sim = np.mean(random_sims)

    _, p_val = mannwhitneyu(driver_sims, random_sims, alternative="greater")
    fold_change = mean_driver_sim / (mean_random_sim + 1e-8)

    print(f"\n{style.title}")
    print(f"  ↳ Driver Compounds Evaluated : {k_drivers} unique molecules")
    print(f"  ↳ {style.similarity_label}: Drivers = {mean_driver_sim:.4f} | Background = {mean_random_sim:.4f}")
    print(f"  ↳ Enrichment Fold Change     : {fold_change:.2f}x (Mann-Whitney p = {p_val:.2e})")

    if p_val < 0.01 and fold_change > 1.02:
        verdict = style.verdict_distinct
    elif p_val < 0.05:
        verdict = style.verdict_weak
    else:
        verdict = style.verdict_diverse
    print(f"  ↳ Structural Verdict         : {verdict}")
    return top_drivers


def report_driver_chemistry(
    bio_scores_df: pd.DataFrame,
    axis_name: str,
    reference_df: pd.DataFrame,
    reference_features: Sequence[str],
    merge_key: str,
    smiles_parquet: str,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Test whether an axis's drivers are alike in bioactivity space *and* in 2D chemical structure.

    Like :func:`report_driver_similarity` with the Chemical Checker, and additionally the pairwise
    ECFP4 Tanimoto similarity of the drivers. The two together give a verdict (see
    :func:`model_benchmarking.chemistry.chemistry_verdict`): a congeneric series, phenotypic
    convergence of different scaffolds, weak clustering, or no structure.

    Parameters
    ----------
    bio_scores_df, axis_name, reference_df, reference_features, merge_key, device
        As in :func:`report_driver_similarity`.
    smiles_parquet : str
        InChIKey -> SMILES table from :func:`model_benchmarking.chemistry.ensure_smiles_lookup`.

    Returns
    -------
    pandas.DataFrame
        The driver compounds, highest score first.
    """
    device = get_device() if device is None else device
    merged, top_drivers = select_driver_compounds(bio_scores_df, axis_name, reference_df, merge_key)
    if len(merged) < 5:
        print("\n🧪 CHEMICAL SPACE: Insufficient compound overlap (< 5 compounds) for network clustering.")
        return top_drivers

    k_drivers = len(top_drivers)
    drivers_tensor = torch.tensor(top_drivers[reference_features].values.astype(np.float32), device=device)
    random_group = merged.sample(n=k_drivers, random_state=42)
    random_tensor = torch.tensor(random_group[reference_features].values.astype(np.float32), device=device)

    driver_sims = get_pairwise_sims(drivers_tensor)
    random_sims = get_pairwise_sims(random_tensor)
    mean_driver_sim = np.mean(driver_sims)
    mean_random_sim = np.mean(random_sims)
    _, p_val = mannwhitneyu(driver_sims, random_sims, alternative="greater")
    fold_change = mean_driver_sim / (mean_random_sim + 1e-8)

    mean_tan, med_tan, max_tan, n_valid = driver_tanimoto(top_drivers[merge_key].tolist(), smiles_parquet)

    print("\n🧪 CHEMICAL SPACE & 2D STRUCTURE:")
    print(f"  ↳ Driver Compounds Evaluated : {k_drivers} unique molecules ({n_valid} resolved to SMILES)")
    print(
        f"  ↳ Mean CC Bioactivity Cosine : Drivers = {mean_driver_sim:.4f} | Background = {mean_random_sim:.4f} "
        f"({fold_change:.2f}x, p = {p_val:.2e})"
    )
    print(f"  ↳ Pairwise ECFP4 Tanimoto    : Mean = {mean_tan:.3f} | Median = {med_tan:.3f} | Max = {max_tan:.3f}")
    print(f"  ↳ Pharmacological Verdict    : {chemistry_verdict(p_val, mean_tan)}")
    return top_drivers


def centered_unit_vectors(X: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
    """Mean-center the rows' population and scale each row to unit length.

    Cosine similarity of the returned vectors equals the Pearson correlation of the original
    profiles across features after removing the average profile. Without centering, embeddings
    that all point in a similar direction look similar to each other whatever the compound does.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_compounds, n_features)
    device : torch.device, optional
        Defaults to :func:`model_benchmarking.utils.get_device`.

    Returns
    -------
    torch.Tensor, shape (n_compounds, n_features)
    """
    device = get_device() if device is None else device
    tensor = torch.tensor(np.asarray(X), device=device, dtype=torch.float32)
    return torch.nn.functional.normalize(tensor - tensor.mean(dim=0, keepdim=True), p=2, dim=1)


def report_morphology_coherence(
    driver_keys: Sequence[str],
    normed_morphology: torch.Tensor,
    key_to_index: Mapping[str, int],
    n_background_draws: int = 12,
    seed: int = 42,
) -> dict | None:
    """Test whether an axis's driver compounds also look alike in morphology.

    Pairwise cosine similarities of the drivers' (centered) morphology profiles are compared with
    those of random groups of the same size drawn from the other compounds (one-sided
    Mann-Whitney test), and the driver closest to the group's centroid is reported as its
    morphological medoid.

    Parameters
    ----------
    driver_keys : sequence of str
        InChIKeys of the driver compounds.
    normed_morphology : torch.Tensor, shape (n_compounds, n_features)
        Output of :func:`centered_unit_vectors`.
    key_to_index : mapping of str to int
        InChIKey -> row of ``normed_morphology``.
    n_background_draws : int, default 12
        Number of random same-size groups forming the background distribution.
    seed : int, default 42

    Returns
    -------
    dict or None
        ``None`` if fewer than 3 drivers have a morphology profile. Otherwise the keys
        ``driver_sims``, ``background_sims`` (pairwise similarities), ``mean_driver``,
        ``mean_background``, ``fold_enrichment`` (NaN unless the background mean is positive),
        ``p_value``, ``medoid_key`` and ``verdict``.

    Notes
    -----
    Pairwise similarities are not independent observations, so the p-value is optimistic.
    """
    driver_idx = [key_to_index[k] for k in driver_keys if k in key_to_index]
    if len(driver_idx) < 3:
        return None

    device = normed_morphology.device
    n_d = len(driver_idx)
    driver_vectors = normed_morphology[driver_idx]
    pair_idx = np.triu_indices(n_d, k=1)
    driver_sims = (driver_vectors @ driver_vectors.T).cpu().numpy()[pair_idx]

    driver_set = set(driver_idx)
    background_pool = torch.tensor([i for i in key_to_index.values() if i not in driver_set], device=device)
    gen = torch.Generator().manual_seed(seed)
    background_sims = []
    for _ in range(n_background_draws):
        chosen = background_pool[torch.randperm(len(background_pool), generator=gen)[:n_d].to(device)]
        sub = normed_morphology[chosen]
        background_sims.extend((sub @ sub.T).cpu().numpy()[pair_idx])
    background_sims = np.array(background_sims)

    mean_driver = float(np.mean(driver_sims))
    mean_background = float(np.mean(background_sims))
    fold = mean_driver / mean_background if mean_background > 0 else float("nan")
    _, p_val = mannwhitneyu(driver_sims, background_sims, alternative="greater")

    driver_np = driver_vectors.cpu().numpy()
    centroid = driver_np.mean(axis=0, keepdims=True)
    cos_to_centroid = (driver_np @ (centroid / (np.linalg.norm(centroid) + 1e-12)).T).ravel()
    medoid_pos = int(np.argmax(cos_to_centroid))
    medoid_key = [k for k in driver_keys if k in key_to_index][medoid_pos]
    medoid_name, medoid_moa = get_rich_compound_annotation(medoid_key)

    if p_val < 1e-5 and mean_driver >= 0.25:
        verdict = "✅ HIGH COHERENCE: Highly distinct, uniform physical cell shape."
    elif p_val < 0.05:
        verdict = "⚠️ MODERATE COHERENCE: Statistically significant visual overlap."
    else:
        verdict = "❌ PHENOTYPIC DIVERGENCE: Visually dispersed (transcriptional intent failed to execute)."

    print("\n🔬 MORPHOLOGICAL PHENOTYPIC COHERENCE (CENTERED CELL PAINTING QC):")
    print(f"  ↳ Evaluated Drivers          : {n_d} compounds")
    fold_text = f" ({fold:.2f}x)" if np.isfinite(fold) else ""
    print(
        f"  ↳ Mean Cosine Similarity     : Drivers = {mean_driver:.4f} | Background = {mean_background:.4f}{fold_text}"
    )
    print(
        f"  ↳ Median Cosine Similarity   : Drivers = {np.median(driver_sims):.4f} | "
        f"Background = {np.median(background_sims):.4f}"
    )
    print(f"  ↳ Phenotypic Significance    : Mann-Whitney p = {p_val:.2e}")
    print(f"  ↳ Morphological Medoid       : {medoid_name} ({medoid_moa})")
    print(f"     ↳ Medoid InChIKey         : {medoid_key} (Cosine to Centroid: {cos_to_centroid[medoid_pos]:.3f})")
    print(f"  ↳ Morphological QC Verdict   : {verdict}")

    return {
        "driver_sims": driver_sims,
        "background_sims": background_sims,
        "mean_driver": mean_driver,
        "mean_background": mean_background,
        "fold_enrichment": fold,
        "p_value": float(p_val),
        "medoid_key": medoid_key,
        "verdict": verdict,
    }


def plot_morphology_coherence(coherence: dict | None, axis_name: str, ax: plt.Axes) -> None:
    """Draw driver vs. background pairwise morphology similarity (density) on ``ax``.

    Parameters
    ----------
    coherence : dict or None
        Output of :func:`report_morphology_coherence`; ``None`` draws a placeholder message.
    axis_name : str
    ax : matplotlib.axes.Axes
    """
    if coherence is None or len(coherence["driver_sims"]) < 2 or len(coherence["background_sims"]) < 2:
        ax.text(0.5, 0.5, "Insufficient morphological overlap for KDE", ha="center", va="center", fontsize=11)
        ax.set_title(f"Morphological Space QC ({axis_name})", fontsize=11, fontweight="bold")
        return

    sns.kdeplot(
        coherence["background_sims"],
        ax=ax,
        label="Random Background Pairs",
        color="#2B2D42",
        fill=True,
        alpha=0.2,
        linewidth=1.5,
    )
    sns.kdeplot(
        coherence["driver_sims"],
        ax=ax,
        label=f"{axis_name} Driver Pairs",
        color="#2A9D8F",
        fill=True,
        alpha=0.5,
        linewidth=2.2,
    )
    ax.axvline(
        x=coherence["mean_driver"],
        color="#2A9D8F",
        linestyle="--",
        linewidth=1.5,
        label=f"Drivers ({coherence['mean_driver']:.3f})",
    )
    ax.axvline(
        x=coherence["mean_background"],
        color="#2B2D42",
        linestyle=":",
        linewidth=1.2,
        label=f"Background ({coherence['mean_background']:.3f})",
    )
    fold = coherence["fold_enrichment"]
    fold_text = f"Fold: {fold:.2f}x | " if np.isfinite(fold) else ""
    ax.set_title(
        f"Centered Morphological QC ({fold_text}p = {coherence['p_value']:.2e})", fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Pairwise Centered Cosine Similarity (Pearson r)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.4)


def print_driver_annotations(top_drivers: pd.DataFrame, axis_name: str, merge_key: str) -> None:
    """Annotate the five top driver compounds and report a shared mechanism if one exists.

    Notes
    -----
    If ChEMBL returns a mechanism record whose ``mechanism_of_action`` is empty, the
    annotation is ``None`` and formatting it raises ``TypeError`` (see
    ``docs/code_review.md``).
    """
    top_keys = top_drivers.head(5)[merge_key].tolist()
    top_scores = top_drivers.head(5)[axis_name].tolist()

    print("\n  ↳ Top 5 Driving Compounds (Auto-Annotated via PubChem / ChEMBL):")
    moa_list = []

    for rank, (ikey, sc) in enumerate(zip(top_keys, top_scores, strict=True), 1):
        name, moa = get_rich_compound_annotation(ikey)
        if moa != "Unannotated":
            moa_list.append(moa)
        print(f"     {rank}. {name:<22} | {moa:<38} | Score: {sc:+.3f} | {ikey}")

    if moa_list:
        most_common_moa, count = Counter(moa_list).most_common(1)[0]
        if count >= 2:
            print(f"  🎯 CONSENSUS MECHANISM FOUND: '{most_common_moa}' ({count}/{len(top_keys)} compounds share this)")
        else:
            print(f"  💡 TARGET DIVERSITY: Multiple distinct targets annotated ({', '.join(set(moa_list))})")
    else:
        print("  ⚠️ NO CURATED TARGETS: Compounds may be novel unannotated screening hits.")


def plot_volcano(gene_df: pd.DataFrame, title: str, title_fontsize: float = 13, ax: plt.Axes | None = None) -> None:
    """Volcano plot of gene-axis correlations (|r| > 0.2 and p < 0.01 highlighted).

    Parameters
    ----------
    gene_df : pandas.DataFrame
        Output of :func:`gene_axis_statistics`.
    title : str
    title_fontsize : float, default 13
    ax : matplotlib.axes.Axes, optional
        Draw into this axes (the caller shows the figure). By default a new figure is created
        and shown.
    """
    standalone = ax is None
    if standalone:
        plt.figure(figsize=(9, 4.5), dpi=100)
        sns.set_style("whitegrid")
        ax = plt.gca()
    r_thresh, p_thresh = 0.20, 0.01
    p_log_thresh = -np.log10(p_thresh)

    up_sig = gene_df[(gene_df["Correlation"] > r_thresh) & (gene_df["P_Value"] < p_thresh)]
    down_sig = gene_df[(gene_df["Correlation"] < -r_thresh) & (gene_df["P_Value"] < p_thresh)]
    not_sig = gene_df[(np.abs(gene_df["Correlation"]) <= r_thresh) | (gene_df["P_Value"] >= p_thresh)]

    ax.scatter(not_sig["Correlation"], not_sig["Neg_Log10_P"], color="gray", alpha=0.2, s=10, label="Not Sig")
    ax.scatter(
        down_sig["Correlation"],
        down_sig["Neg_Log10_P"],
        color="#1D3557",
        alpha=0.6,
        s=20,
        label=f"Down (r < -{r_thresh})",
    )
    ax.scatter(
        up_sig["Correlation"], up_sig["Neg_Log10_P"], color="#E63946", alpha=0.6, s=20, label=f"Up (r > +{r_thresh})"
    )

    n_labels = 5 if standalone else 4
    top_label = pd.concat(
        [
            up_sig.sort_values(by="Correlation", ascending=False, kind="stable").head(n_labels),
            down_sig.sort_values(by="Correlation", ascending=True, kind="stable").head(n_labels),
        ]
    )

    for _, r_lbl in top_label.iterrows():
        ax.text(
            r_lbl["Correlation"],
            r_lbl["Neg_Log10_P"],
            r_lbl["Gene_Symbol"],
            fontsize=8,
            fontweight="bold",
            ha="left" if r_lbl["Correlation"] > 0 else "right",
            bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 1},
        )

    ax.axvline(x=r_thresh, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.axvline(x=-r_thresh, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.axhline(y=p_log_thresh, color="black", linestyle="--", linewidth=0.8, alpha=0.4)

    max_c = max(np.abs(gene_df["Correlation"].min()), np.abs(gene_df["Correlation"].max()))
    ax.set_xlim(-max_c - 0.08, max_c + 0.08)
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=10)
    ax.set_xlabel("Pearson Correlation ($r$)", fontsize=11)
    ax.set_ylabel("Significance ($-\\log_{10}$ P)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    if standalone:
        plt.tight_layout()
        plt.show()
