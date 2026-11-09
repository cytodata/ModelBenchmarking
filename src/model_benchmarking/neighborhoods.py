"""Test whether a group of compounds sits close together in a representation.

Used to ask, for a set of "driver" compounds that score highly on one shared
morphology-transcriptomics axis: are these compounds also each other's nearest
neighbours in chemical-structure space? In bioactivity space?
"""

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import hypergeom

from model_benchmarking.utils import get_device


def get_pairwise_sims(tensor: torch.Tensor) -> np.ndarray:
    """All pairwise cosine similarities within a set of vectors.

    Parameters
    ----------
    tensor : torch.Tensor, shape (k, n_features)

    Returns
    -------
    numpy.ndarray, shape (k * (k - 1) / 2,)
        Upper-triangle (i < j) cosine similarities.
    """
    normed = torch.nn.functional.normalize(tensor, p=2, dim=1)
    sim_mat = torch.matmul(normed, normed.T)
    k = tensor.shape[0]
    triu_indices = torch.triu_indices(k, k, offset=1, device=tensor.device)
    return sim_mat[triu_indices[0], triu_indices[1]].cpu().numpy()


def evaluate_driver_neighborhoods(
    driver_inchikeys: Iterable[str],
    all_inchikeys: Sequence[str] | np.ndarray,
    modalities_dict: Mapping[str, np.ndarray],
    k_neighbors: int = 10,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Neighbourhood enrichment and retrieval precision of a compound set.

    For every representation, a k-nearest-neighbour graph (cosine similarity)
    is built over all compounds. Two scores are then computed for the driver
    set:

    * **Fold enrichment / hypergeometric p-value**: how many kNN edges connect
      two drivers, compared with the number expected if edges were placed at
      random.
    * **mAP** (mean average precision): when ranking all compounds by
      similarity to a driver, how early do the other drivers appear?

    Parameters
    ----------
    driver_inchikeys : iterable of str
    all_inchikeys : array-like of str
        Row labels for every matrix in ``modalities_dict``.
    modalities_dict : dict of {str: numpy.ndarray}
        Representation name -> matrix whose rows follow ``all_inchikeys``.
    k_neighbors : int, default 10
    device : torch.device, optional
        Defaults to :func:`model_benchmarking.utils.get_device`.

    Returns
    -------
    pandas.DataFrame
        One row per representation.

    Notes
    -----
    Self-similarity is set to -1 before ranking, so each driver appears as its
    own (last-ranked) relevant hit in the average-precision calculation. This
    slightly lowers every mAP value.
    """
    device = get_device() if device is None else device
    driver_set = set(driver_inchikeys)
    N = len(all_inchikeys)
    n_drivers = len(driver_set)

    is_driver = np.array([ikey in driver_set for ikey in all_inchikeys])
    driver_indices = np.where(is_driver)[0]

    total_possible_pairs = N * (N - 1) // 2
    driver_possible_pairs = n_drivers * (n_drivers - 1) // 2

    results = []

    for mod_name, embs_np in modalities_dict.items():
        # 1. Pairwise Cosine Similarity on GPU
        embs_t = torch.tensor(embs_np, device=device, dtype=torch.float32)
        normed = torch.nn.functional.normalize(embs_t, p=2, dim=1)
        sim_matrix_t = torch.matmul(normed, normed.T)

        # Zero-out self-similarity
        sim_matrix_t.fill_diagonal_(-1.0)

        # Extract top k neighbors per compound
        _, top_k_indices_t = torch.topk(sim_matrix_t, k=k_neighbors, dim=1)
        top_k_indices = top_k_indices_t.cpu().numpy()
        sim_matrix = sim_matrix_t.cpu().numpy()

        # 2. Part IV.2: Deduplicated Undirected Graph Transform
        sources = np.repeat(np.arange(N), k_neighbors)
        targets = top_k_indices.flatten()

        # Force min on left, max on right to shatter A->B vs B->A double-counting
        edge_left = np.minimum(sources, targets)
        edge_right = np.maximum(sources, targets)
        unique_edges = np.unique(np.column_stack([edge_left, edge_right]), axis=0)
        total_unique_edges = len(unique_edges)

        # Count edges where BOTH endpoints are driver molecules
        left_is_driver = is_driver[unique_edges[:, 0]]
        right_is_driver = is_driver[unique_edges[:, 1]]
        observed_driver_edges = np.sum(left_is_driver & right_is_driver)

        # Expected edges by chance
        network_density = total_unique_edges / total_possible_pairs
        expected_driver_edges = driver_possible_pairs * network_density
        fold_enrichment = observed_driver_edges / (expected_driver_edges + 1e-12)

        # Hypergeometric Survival Function: P(X >= observed)
        p_val_hypergeom = hypergeom.sf(
            observed_driver_edges - 1, total_possible_pairs, total_unique_edges, driver_possible_pairs
        )

        # 3. Part IV.3: Mean Average Precision (mAP)
        ap_scores = []
        for d_idx in driver_indices:
            sims = sim_matrix[d_idx]
            ranked_indices = np.argsort(-sims)
            relevance = is_driver[ranked_indices]

            cum_hits = np.cumsum(relevance)
            ranks = np.arange(1, len(relevance) + 1)
            precisions = cum_hits / ranks

            if np.sum(relevance) > 0:
                ap = np.sum(precisions * relevance) / np.sum(relevance)
                ap_scores.append(ap)

        map_score = np.mean(ap_scores)

        results.append(
            {
                "Representation Space": mod_name,
                "Internal Edges": int(observed_driver_edges),
                "Expected": float(expected_driver_edges),
                "Fold Enrichment": fold_enrichment,
                "Hypergeom p-val": p_val_hypergeom,
                "mAP Score": map_score,
            }
        )

    return pd.DataFrame(results)
