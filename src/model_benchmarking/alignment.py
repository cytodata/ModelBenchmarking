"""Measure how well two representations of the same compounds agree.

Two complementary views are implemented:

* **Linear CKA** (Centered Kernel Alignment) gives one number between 0 and 1
  describing how similar the overall geometry of two representations is: do
  compounds that look alike in space X also look alike in space Y?
* **SVCCA** (Singular Vector Canonical Correlation Analysis) first reduces each
  representation to its main principal components, then finds pairs of
  directions ("canonical axes") that are maximally correlated between the two.
  The sorted canonical correlations form a *decay curve*: how many independent
  axes of shared variation exist, and how strong each one is.

Both are compared against a permutation null in which the pairing of compounds
between the two spaces is shuffled, so that any "alignment" left over is what
chance alone produces for data of this size and dimensionality.

In every function, rows of ``X`` and ``Y`` must refer to the same compounds in
the same order.
"""

import numpy as np
import numpy.typing as npt
import torch

from model_benchmarking.utils import get_device


# ---------------------------------------------------------------------------
# NumPy implementation (used for the pilot analysis)
# ---------------------------------------------------------------------------
def linear_cka(X: np.ndarray, Y: np.ndarray, centered: bool = True) -> float:
    """Linear Centered Kernel Alignment between two matrices.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_compounds, n_features_x)
    Y : numpy.ndarray, shape (n_compounds, n_features_y)
    centered : bool, default True
        Subtract the column means before computing CKA.

    Returns
    -------
    float
        CKA in [0, 1]; 1 means identical geometry up to rotation and scaling.
    """
    if centered:
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)

    dot_prod = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    norm_x = np.linalg.norm(X.T @ X, ord="fro")
    norm_y = np.linalg.norm(Y.T @ Y, ord="fro")

    # Added + 1e-8 to prevent Division by Zero if a matrix is entirely empty
    return dot_prod / (norm_x * norm_y + 1e-8)


def evaluate_alignment(X: np.ndarray, Y: np.ndarray, n_permutations: int = 100) -> tuple[float, float, float, float]:
    """Linear CKA with a row-permutation null baseline (NumPy).

    Parameters
    ----------
    X, Y : numpy.ndarray
        Paired representations (same compounds, same row order).
    n_permutations : int, default 100
        Number of shuffles of the rows of ``Y`` used to build the null.

    Returns
    -------
    true_cka : float
    null_mean : float
        Mean CKA over shuffled pairings.
    adj_signal : float
        ``true_cka - null_mean``.
    p_value : float
        Empirical one-sided p-value ``(#null >= true + 1) / (n_permutations + 1)``;
        the smallest attainable value is ``1 / (n_permutations + 1)``.

    Notes
    -----
    Shuffles use NumPy's global random state (``np.random.permutation``).
    """
    print(f"\nComputing True CKA between {X.shape[1]}D Morphology and {Y.shape[1]}D Transcriptomics...")
    true_cka = linear_cka(X, Y, centered=True)

    print(f"Running {n_permutations} permutation scrambles to establish null baseline...")
    null_ckas = []
    N = Y.shape[0]

    for _ in range(n_permutations):
        shuffled_indices = np.random.permutation(N)
        Y_shuffled = Y[shuffled_indices, :]
        null_ckas.append(linear_cka(X, Y_shuffled, centered=True))

    null_ckas = np.array(null_ckas)
    null_mean = np.mean(null_ckas)

    adj_signal = true_cka - null_mean
    p_value = (np.sum(null_ckas >= true_cka) + 1) / (n_permutations + 1)

    return true_cka, null_mean, adj_signal, p_value


# ---------------------------------------------------------------------------
# PyTorch implementation (exact closed form, GPU-accelerated when available)
# ---------------------------------------------------------------------------
def svd_flip_torch(u: torch.Tensor, v: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Give singular/eigen-vectors a deterministic sign.

    Eigen- and singular vectors are only defined up to a sign flip, which makes
    results differ between runs and hardware. Each column of ``u`` is flipped so
    that its largest-magnitude entry is positive (the same convention as
    scikit-learn's ``svd_flip``).

    Parameters
    ----------
    u : torch.Tensor, shape (n, k)
    v : torch.Tensor, shape (k', m), optional
        If given, its first ``min(k, k')`` rows are flipped consistently with
        ``u``. **Modified in place.**

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        ``u`` alone, or ``(u, v)`` when ``v`` is given.
    """
    max_abs_cols = torch.argmax(torch.abs(u), dim=0)
    signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
    signs[signs == 0] = 1.0
    u = u * signs

    if v is not None:
        # Match only the shared canonical components
        k_shared = min(u.shape[1], v.shape[0])
        v[:k_shared] = v[:k_shared] * signs[:k_shared].unsqueeze(1)
        return u, v
    return u


def get_exact_subspace_basis(M: torch.Tensor, k: int = 50) -> torch.Tensor:
    """Orthonormal basis of the top-``k`` principal components of ``M``.

    Uses the eigendecomposition of the ``n x n`` Gram matrix ``M @ M.T``, which
    is fast when there are fewer compounds than features. The returned columns
    are the left singular vectors of ``M`` (i.e. normalized PC scores).

    Parameters
    ----------
    M : torch.Tensor, shape (n_compounds, n_features)
        Should already be column-centered.
    k : int, default 50
        Number of components to keep.

    Returns
    -------
    torch.Tensor, shape (n_compounds, k)
    """
    G = torch.matmul(M, M.T)
    eigvals, eigvecs = torch.linalg.eigh(G)
    idx = torch.argsort(eigvals, descending=True)[:k]
    U_k = eigvecs[:, idx]
    U_k = svd_flip_torch(U_k)
    return U_k


def get_exact_subspace_basis_var_thresh(
    M: torch.Tensor, variance_threshold: float = 0.90, max_rank_cap: int | None = None
) -> tuple[torch.Tensor, int, float]:
    """Like :func:`get_exact_subspace_basis`, but choose ``k`` by explained variance.

    Keeps the smallest number of principal components whose cumulative
    explained variance reaches ``variance_threshold``, capped at
    ``max_rank_cap`` and at the matrix rank, with a floor of 5 components.

    Parameters
    ----------
    M : torch.Tensor, shape (n_compounds, n_features)
        Should already be column-centered.
    variance_threshold : float, default 0.90
    max_rank_cap : int, optional

    Returns
    -------
    U_k : torch.Tensor, shape (n_compounds, k)
    k_needed : int
        Number of components kept.
    actual_var : float
        Cumulative explained variance of the kept components.
    """
    G = torch.matmul(M, M.T)
    eigvals, eigvecs = torch.linalg.eigh(G)

    idx_all = torch.argsort(eigvals, descending=True)
    eigvals_desc = eigvals[idx_all]
    eigvecs_desc = eigvecs[:, idx_all]

    tot_var = torch.sum(torch.clamp(eigvals_desc, min=0.0))
    cum_var = torch.cumsum(torch.clamp(eigvals_desc, min=0.0), dim=0) / (tot_var + 1e-12)

    k_needed = int((cum_var < variance_threshold).sum().item()) + 1

    if max_rank_cap is not None:
        k_needed = min(k_needed, max_rank_cap)
    k_needed = min(k_needed, M.shape[1], M.shape[0])
    k_needed = max(k_needed, 5)

    U_k = eigvecs_desc[:, :k_needed]
    U_k = svd_flip_torch(U_k)

    actual_var = float(cum_var[min(k_needed - 1, len(cum_var) - 1)].item())
    return U_k, k_needed, actual_var


def compress_to_exact_dimension(
    X_np: np.ndarray,
    target_dim: int = 384,
    dtype: npt.DTypeLike = np.float32,
    device: torch.device | None = None,
) -> np.ndarray:
    """Reduce a representation to at most ``target_dim`` dimensions with PCA.

    Representations with more than ``target_dim`` features are projected onto
    their top principal components (scores ``U * Sigma``), so that models with
    very different embedding sizes can be compared on an equal footing.
    Representations that are already small enough are returned unchanged.

    Parameters
    ----------
    X_np : numpy.ndarray, shape (n_compounds, n_features)
    target_dim : int, default 384
    dtype : numpy dtype, default numpy.float32
        dtype of the returned array.
    device : torch.device, optional
        Defaults to :func:`model_benchmarking.utils.get_device`.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_compounds, n_features)`` if ``n_features <= target_dim``,
        otherwise ``(n_compounds, min(n_compounds, target_dim))``.

    Notes
    -----
    Components are computed from the ``n x n`` Gram matrix, so at most
    ``n_compounds`` of them exist. After centering, the last one carries no
    variance (its eigenvalue is clamped to 1e-8).
    """
    device = get_device() if device is None else device
    if X_np.shape[1] <= target_dim:
        return X_np.astype(dtype)

    X = torch.tensor(X_np, device=device, dtype=torch.float64)
    X_c = X - X.mean(dim=0)

    # Gram matrix eigendecomposition (N x N)
    G = torch.matmul(X_c, X_c.T)
    eigvals, eigvecs = torch.linalg.eigh(G)

    idx = torch.argsort(eigvals, descending=True)[:target_dim]
    top_eigvals = torch.clamp(eigvals[idx], min=1e-8)
    top_eigvecs = eigvecs[:, idx]

    top_eigvecs = svd_flip_torch(top_eigvecs)

    # Exact PCA projection: U * Sigma
    singular_values = torch.sqrt(top_eigvals)
    X_compressed = (top_eigvecs * singular_values).cpu().numpy().astype(dtype)
    return X_compressed


def evaluate_alignment_exact(
    X_np: np.ndarray,
    Y_np: np.ndarray,
    n_permutations: int = 100,
    seed: int = 42,
    device: torch.device | None = None,
) -> tuple[float, float, float, float]:
    """Linear CKA with a permutation null, computed from Gram matrices (PyTorch).

    Equivalent to :func:`evaluate_alignment` but uses the ``n x n`` kernel
    formulation, which is faster when features outnumber compounds, and a
    seeded generator so results are reproducible.

    Parameters
    ----------
    X_np, Y_np : numpy.ndarray
        Paired representations (same compounds, same row order).
    n_permutations : int, default 100
    seed : int, default 42
    device : torch.device, optional
        Defaults to :func:`model_benchmarking.utils.get_device`.

    Returns
    -------
    true_cka, null_mean, adj_sig, p_val : float
        See :func:`evaluate_alignment`. With 100 permutations the smallest
        attainable p-value is 1/101 ≈ 0.0099.
    """
    device = get_device() if device is None else device
    X = torch.tensor(X_np, device=device, dtype=torch.float64)
    Y = torch.tensor(Y_np, device=device, dtype=torch.float64)
    N = X.shape[0]

    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)

    K = torch.matmul(X, X.T)
    L = torch.matmul(Y, Y.T)

    norm_x = torch.sqrt(torch.sum(K * K))
    norm_y = torch.sqrt(torch.sum(L * L))
    denom = (norm_x * norm_y) + 1e-12

    true_cka = (torch.sum(K * L) / denom).item()

    gen = torch.Generator(device=device).manual_seed(seed)
    null_ckas = []

    for _ in range(n_permutations):
        idx = torch.randperm(N, generator=gen, device=device)
        L_perm = L[idx][:, idx]
        null_ckas.append((torch.sum(K * L_perm) / denom).item())

    null_ckas = np.array(null_ckas)
    null_mean = np.mean(null_ckas)
    adj_sig = true_cka - null_mean
    p_val = (np.sum(null_ckas >= true_cka) + 1) / (n_permutations + 1)

    return true_cka, null_mean, adj_sig, p_val


def get_svcca_decay_exact(
    X_np: np.ndarray, Y_np: np.ndarray, k_dims: int = 50, device: torch.device | None = None
) -> np.ndarray:
    """SVCCA canonical correlations using a fixed number of principal components.

    Parameters
    ----------
    X_np, Y_np : numpy.ndarray
        Paired representations (same compounds, same row order).
    k_dims : int, default 50
        Maximum number of PCs per representation; also capped at 80% of the
        number of compounds and at each representation's dimensionality.
    device : torch.device, optional
        Defaults to :func:`model_benchmarking.utils.get_device`.

    Returns
    -------
    numpy.ndarray
        Canonical correlations sorted from strongest to weakest.
    """
    device = get_device() if device is None else device
    X = torch.tensor(X_np, device=device, dtype=torch.float64)
    Y = torch.tensor(Y_np, device=device, dtype=torch.float64)
    N = X.shape[0]
    max_k = min(int(N * 0.8), X.shape[1], Y.shape[1], k_dims)

    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)

    Qa = get_exact_subspace_basis(X, max_k)
    Qb = get_exact_subspace_basis(Y, max_k)

    cross_cov = torch.matmul(Qa.T, Qb)
    U, S, Vt = torch.linalg.svd(cross_cov)
    U, _ = svd_flip_torch(U, Vt)

    return torch.sort(S, descending=True)[0].cpu().numpy()


def get_svcca_with_model_null_exact(
    X_np: np.ndarray,
    Y_np: np.ndarray,
    var_thresh: float = 0.90,
    n_perms: int = 100,
    seed: int = 42,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, float, float]:
    """SVCCA on variance-threshold subspaces, with a model-specific null.

    Each representation is reduced to the PCs explaining ``var_thresh`` of its
    variance (see :func:`get_exact_subspace_basis_var_thresh`). The null is
    built by shuffling compound pairings ``n_perms`` times for *this* pair of
    matrices, so it accounts for the model's own dimensionality.

    Parameters
    ----------
    X_np, Y_np : numpy.ndarray
        Paired representations (same compounds, same row order).
    var_thresh : float, default 0.90
    n_perms : int, default 100
    seed : int, default 42
    device : torch.device, optional
        Defaults to :func:`model_benchmarking.utils.get_device`.

    Returns
    -------
    true_corrs : numpy.ndarray
        Canonical correlations (length ``min(k_x, k_y)``), strongest first.
    null_mean, null_std : numpy.ndarray
        Mean and standard deviation of the shuffled canonical correlations at
        each rank.
    sig_axes : int
        Number of ranks where ``true > null_mean + 2 * null_std``.
    k_x, k_y : int
        Number of PCs kept for X and Y.
    var_x_act, var_y_act : float
        Variance explained by the kept PCs.

    Notes
    -----
    Because ``k_x`` and ``k_y`` differ between models, the number of canonical
    axes differs too; averages over the decay curve are then taken over
    different numbers of axes.
    """
    device = get_device() if device is None else device
    X = torch.tensor(X_np, device=device, dtype=torch.float64)
    Y = torch.tensor(Y_np, device=device, dtype=torch.float64)
    N = X.shape[0]
    rank_cap = int(N * 0.8)

    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)

    Qa, k_x, var_x_act = get_exact_subspace_basis_var_thresh(X, variance_threshold=var_thresh, max_rank_cap=rank_cap)
    Qb, k_y, var_y_act = get_exact_subspace_basis_var_thresh(Y, variance_threshold=var_thresh, max_rank_cap=rank_cap)

    # Compute compact thin SVD on the rectangular cross-covariance matrix
    cross_cov = torch.matmul(Qa.T, Qb)  # Shape: (k_x x k_y)
    U, S, Vt = torch.linalg.svd(cross_cov, full_matrices=False)
    U, _ = svd_flip_torch(U, Vt)
    true_corrs = torch.sort(S, descending=True)[0].cpu().numpy()

    # Model-specific permutations (row slicing of the fixed basis)
    gen = torch.Generator(device=device).manual_seed(seed)
    null_corrs_list = []

    for _ in range(n_perms):
        idx = torch.randperm(N, generator=gen, device=device)
        cross_cov_null = torch.matmul(Qa.T, Qb[idx])
        S_null = torch.linalg.svdvals(cross_cov_null)
        null_corrs_list.append(torch.sort(S_null, descending=True)[0].cpu().numpy())

    null_matrix = np.array(null_corrs_list)
    null_mean = null_matrix.mean(axis=0)
    null_std = null_matrix.std(axis=0)

    sig_axes = int(np.sum(true_corrs > (null_mean + (2 * null_std))))

    return true_corrs, null_mean, null_std, sig_axes, k_x, k_y, var_x_act, var_y_act


def get_svcca_decay_and_null_exact(
    X_np: np.ndarray,
    Y_np: np.ndarray,
    k_dims: int = 50,
    n_perms: int = 100,
    seed: int = 42,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """SVCCA with a fixed number of PCs, plus a model-specific permutation null.

    Same as :func:`get_svcca_decay_exact` but also returns the null
    distribution, as in :func:`get_svcca_with_model_null_exact`.

    Parameters
    ----------
    X_np, Y_np : numpy.ndarray
        Paired representations (same compounds, same row order).
    k_dims : int, default 50
    n_perms : int, default 100
    seed : int, default 42
    device : torch.device, optional
        Defaults to :func:`model_benchmarking.utils.get_device`.

    Returns
    -------
    true_corrs, null_mean, null_std : numpy.ndarray
    sig_axes : int
    """
    device = get_device() if device is None else device
    X = torch.tensor(X_np, device=device, dtype=torch.float64)
    Y = torch.tensor(Y_np, device=device, dtype=torch.float64)
    N = X.shape[0]
    max_k = min(int(N * 0.8), X.shape[1], Y.shape[1], k_dims)

    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)

    Qa = get_exact_subspace_basis(X, max_k)
    Qb = get_exact_subspace_basis(Y, max_k)

    cross_cov = torch.matmul(Qa.T, Qb)
    U, S, Vt = torch.linalg.svd(cross_cov, full_matrices=False)
    U, _ = svd_flip_torch(U, Vt)
    true_corrs = torch.sort(S, descending=True)[0].cpu().numpy()

    # Model-specific permutations (row slicing of the fixed basis)
    gen = torch.Generator(device=device).manual_seed(seed)
    null_corrs_list = []

    for _ in range(n_perms):
        idx = torch.randperm(N, generator=gen, device=device)
        cross_cov_null = torch.matmul(Qa.T, Qb[idx])
        S_null = torch.linalg.svdvals(cross_cov_null)
        null_corrs_list.append(torch.sort(S_null, descending=True)[0].cpu().numpy())

    null_matrix = np.array(null_corrs_list)
    null_mean = null_matrix.mean(axis=0)
    null_std = null_matrix.std(axis=0)
    sig_axes = int(np.sum(true_corrs > (null_mean + (2 * null_std))))

    return true_corrs, null_mean, null_std, sig_axes


def reduce_dimensions(
    X_np: np.ndarray,
    mode: str = "variance",
    var_threshold: float = 0.99,
    fixed_dims: int = 384,
    device: torch.device | None = None,
) -> np.ndarray:
    """Optionally compress a representation with PCA before it is compared.

    Parameters
    ----------
    X_np : numpy.ndarray, shape (n_compounds, n_features)
    mode : {"variance", "fixed", "none"}, default "variance"
        ``"none"`` returns the data unchanged; ``"fixed"`` keeps ``fixed_dims`` components;
        ``"variance"`` keeps the fewest components explaining ``var_threshold`` of the variance.
    var_threshold : float, default 0.99
        Used when ``mode == "variance"``.
    fixed_dims : int, default 384
        Used when ``mode == "fixed"``.
    device : torch.device, optional
        Defaults to :func:`model_benchmarking.utils.get_device`.

    Returns
    -------
    numpy.ndarray
        PCA scores (``U * Sigma``), float32, with at most 80% as many components as compounds
        (or the unchanged data for ``mode="none"``).

    Notes
    -----
    Whichever dimension is smaller (compounds or features) is decomposed: a thin SVD of the
    data, or the eigendecomposition of the ``n x n`` Gram matrix.
    """
    device = get_device() if device is None else device
    if mode == "none":
        return X_np.astype(np.float32)
    if mode not in ("variance", "fixed"):
        raise ValueError(f"mode must be 'variance', 'fixed' or 'none', received: {mode!r}")

    n_samples, n_features = X_np.shape
    X = torch.tensor(X_np, device=device, dtype=torch.float32)
    X_c = X - X.mean(dim=0)
    max_k = min(n_features, int(n_samples * 0.8))

    if n_samples > n_features:
        U, S, _ = torch.linalg.svd(X_c, full_matrices=False)
        U = svd_flip_torch(U)
        variance = S**2
        scores = U * S
    else:
        eigvals, eigvecs = torch.linalg.eigh(torch.matmul(X_c, X_c.T))
        order = torch.argsort(eigvals, descending=True)
        variance = torch.clamp(eigvals[order], min=1e-8)
        scores = svd_flip_torch(eigvecs[:, order]) * torch.sqrt(variance)

    if mode == "variance":
        cum_variance = torch.cumsum(variance, dim=0) / torch.sum(variance)
        k = int(torch.argmax((cum_variance >= var_threshold).to(torch.int64)).item()) + 1
        k = min(k, max_k)
    else:
        k = min(fixed_dims, max_k)

    return scores[:, :k].cpu().numpy().astype(np.float32)


def get_svcca_decay_with_cutoff_exact(
    X_np: np.ndarray,
    Y_np: np.ndarray,
    k_dims: int = 50,
    n_perms: int = 100,
    seed: int = 42,
    thresh_mode: str = "std",
    sigma_val: float = 3.0,
    pct_val: float = 95.0,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """SVCCA with a model-specific null and a configurable significance cut-off.

    Like :func:`get_svcca_decay_and_null_exact`, but the noise floor that an axis must exceed
    to count as significant can be chosen, and the whole floor is returned for plotting.

    Parameters
    ----------
    X_np, Y_np : numpy.ndarray
        Paired representations (same compounds, same row order).
    k_dims : int, default 50
    n_perms : int, default 100
    seed : int, default 42
    thresh_mode : {"std", "percentile"}, default "std"
        ``"std"``: null mean + ``sigma_val`` standard deviations at each rank.
        ``"percentile"``: the ``pct_val``-th percentile of the null at each rank.
    sigma_val : float, default 3.0
    pct_val : float, default 95.0
    device : torch.device, optional
        Defaults to :func:`model_benchmarking.utils.get_device`.

    Returns
    -------
    true_corrs, null_mean, null_std, cutoff_curve : numpy.ndarray
        ``cutoff_curve`` is the noise floor at each rank.
    sig_axes : int
        Number of ranks where ``true_corrs > cutoff_curve``.
    """
    if thresh_mode not in ("std", "percentile"):
        raise ValueError(f"thresh_mode must be 'std' or 'percentile', received: {thresh_mode!r}")

    device = get_device() if device is None else device
    X = torch.tensor(X_np, device=device, dtype=torch.float64)
    Y = torch.tensor(Y_np, device=device, dtype=torch.float64)
    N = X.shape[0]
    max_k = min(int(N * 0.8), X.shape[1], Y.shape[1], k_dims)

    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)

    Qa = get_exact_subspace_basis(X, max_k)
    Qb = get_exact_subspace_basis(Y, max_k)

    true_corrs = torch.sort(torch.linalg.svdvals(torch.matmul(Qa.T, Qb)), descending=True)[0].cpu().numpy()

    gen = torch.Generator(device=device).manual_seed(seed)
    null_corrs_list = []
    for _ in range(n_perms):
        idx = torch.randperm(N, generator=gen, device=device)
        S_null = torch.linalg.svdvals(torch.matmul(Qa.T, Qb[idx]))
        null_corrs_list.append(torch.sort(S_null, descending=True)[0].cpu().numpy())

    null_matrix = np.array(null_corrs_list)
    null_mean = null_matrix.mean(axis=0)
    null_std = null_matrix.std(axis=0)

    if thresh_mode == "std":
        cutoff_curve = null_mean + sigma_val * null_std
    else:
        cutoff_curve = np.percentile(null_matrix, pct_val, axis=0)

    sig_axes = int(np.sum(true_corrs > cutoff_curve))
    return true_corrs, null_mean, null_std, cutoff_curve, sig_axes
