"""Compare representations through the overlap of their k-nearest-neighbour graphs.

For each representation, every compound is linked to its ``k`` most similar compounds
(cosine similarity). The links are treated as undirected "edges"; two representations
agree when they share many edges. The number of shared edges is compared with the number
expected if one representation's edges were placed at random.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from model_benchmarking.data import safely_set_inchikey_index


def load_embedding_tables(
    datasets_config: Mapping[str, str], target_keys: set[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Load embedding tables, index them by InChIKey and keep numeric columns only.

    Parameters
    ----------
    datasets_config : mapping of str to str
        Dataset name -> Parquet path (local or ``gs://``).
    target_keys : set of str, optional
        If given, keep only these InChIKeys and skip datasets left with no rows.

    Returns
    -------
    dict of str to pandas.DataFrame
        Dataset name -> float32 table with one row per InChIKey. Datasets that fail to
        load are reported and left out.

    Notes
    -----
    Duplicate InChIKeys (e.g. replicate wells) are reduced to their **first** row, and
    every numeric column is kept as a feature. See ``docs/code_review.md`` (H1, H2).
    """
    loaded_dfs = {}
    for name, path in tqdm(datasets_config.items(), desc="Loading Parquets"):
        try:
            df = pd.read_parquet(path)
            df = safely_set_inchikey_index(df, name)
            if target_keys is not None:
                # Keep ONLY the molecules present in the target subset
                valid_keys = df.index.intersection(target_keys)
                df = df.loc[valid_keys]
            df = df[~df.index.duplicated(keep="first")]

            # Isolate pure numeric features to keep RAM usage low
            df_embs = df.select_dtypes(include=[np.number]).astype(np.float32)
            if target_keys is not None and len(df_embs) == 0:
                print(f"   ⚠️ {name} resulted in 0 matching molecules. It will be skipped.")
                continue
            loaded_dfs[name] = df_embs
        except Exception as e:
            print(f"   ❌ Failed to load {name}: {e}")
    return loaded_dfs


def knn_neighbor_indices(X: np.ndarray, max_k: int) -> np.ndarray:
    """Indices of each row's ``max_k`` nearest neighbours (cosine), with the row itself first.

    Parameters
    ----------
    X : numpy.ndarray, shape (n, d)
    max_k : int

    Returns
    -------
    numpy.ndarray, shape (n, max_k + 1)
    """
    knn = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine", n_jobs=-1).fit(X)
    _, indices = knn.kneighbors(X)
    return indices


def undirected_knn_edges(indices: np.ndarray, k: int) -> set[tuple[int, int]]:
    """Deduplicated, undirected edges linking each row to its first ``k`` neighbours.

    Parameters
    ----------
    indices : numpy.ndarray
        Output of :func:`knn_neighbor_indices` (column 0 is the row itself).
    k : int

    Returns
    -------
    set of (int, int)
        Pairs ``(i, j)`` with ``i < j``; an edge found from both ends is counted once.
    """
    current_rows = np.repeat(np.arange(indices.shape[0]), k)
    cols = indices[:, 1 : k + 1].flatten()
    edge_min, edge_max = np.minimum(current_rows, cols), np.maximum(current_rows, cols)
    return {(u, v) for u, v in zip(edge_min, edge_max, strict=True) if u != v}


def edge_overlap_statistics(
    edges_x: set[tuple[int, int]], edges_y: set[tuple[int, int]], n_compounds: int
) -> dict[str, float | str]:
    """Overlap between two kNN edge sets and its enrichment over chance.

    Parameters
    ----------
    edges_x, edges_y : set of (int, int)
        Edge sets of the two representations (from :func:`undirected_knn_edges`).
    n_compounds : int
        Number of compounds both graphs are built on.

    Returns
    -------
    dict
        ``Overlap_Pct``
            Percentage of X's edges also present in Y.
        ``Random_Overlap``
            Percentage expected by chance: Y's edges / all possible pairs.
        ``Fold_Enrichment``
            Observed / expected overlap.
        ``P_Value``
            Hypergeometric probability of at least this many shared edges by chance.
        ``Likelihood_GT_Chance``
            ``(1 - p) * 100`` as text. This is **not** the probability that the
            association is real.
    """
    total_possible_pairs = (n_compounds * (n_compounds - 1)) / 2
    num_shared = len(edges_x.intersection(edges_y))
    overlap_pct = num_shared / len(edges_x) if len(edges_x) > 0 else 0
    random_expected = len(edges_y) / total_possible_pairs if total_possible_pairs > 0 else 0
    fold_enrichment = overlap_pct / random_expected if random_expected > 0 else 0

    # Statistical likelihood (hypergeometric test)
    drawn = len(edges_x)
    successes_in_pop = len(edges_y)
    if num_shared > 0 and drawn > 0:
        p_val = hypergeom.sf(num_shared - 1, total_possible_pairs, successes_in_pop, drawn)
    else:
        p_val = 1.0

    likelihood_pct = (1.0 - p_val) * 100
    likelihood_str = ">99.99%" if p_val < 1e-4 else f"{likelihood_pct:.2f}%"

    return {
        "Overlap_Pct": overlap_pct * 100,
        "Random_Overlap": random_expected * 100,
        "Fold_Enrichment": fold_enrichment,
        "P_Value": p_val,
        "Likelihood_GT_Chance": likelihood_str,
    }
