"""Tests for model_benchmarking.alignment (CKA, SVCCA and permutation nulls)."""

import numpy as np
import pytest
import torch

from model_benchmarking.alignment import (
    compress_to_exact_dimension,
    evaluate_alignment,
    evaluate_alignment_exact,
    get_exact_subspace_basis,
    get_exact_subspace_basis_var_thresh,
    get_svcca_decay_and_null_exact,
    get_svcca_decay_exact,
    get_svcca_decay_with_cutoff_exact,
    get_svcca_with_model_null_exact,
    linear_cka,
    reduce_dimensions,
    svd_flip_torch,
)

CPU = torch.device("cpu")


@pytest.fixture
def paired_data() -> tuple[np.ndarray, np.ndarray]:
    """Two representations that share a 5-dimensional latent signal."""
    rng = np.random.default_rng(0)
    n = 150
    latent = rng.normal(size=(n, 5))
    X = np.hstack([latent @ rng.normal(size=(5, 40)), 0.5 * rng.normal(size=(n, 20))])
    Y = np.hstack([latent @ rng.normal(size=(5, 30)), 0.5 * rng.normal(size=(n, 10))])
    return X.astype(np.float32), Y.astype(np.float32)


def test_linear_cka_identity_and_invariances(paired_data: tuple[np.ndarray, np.ndarray]) -> None:
    """CKA of a matrix with itself, or with a rotated and scaled copy of itself, is 1."""
    X, _ = paired_data
    assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-6)
    Q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(X.shape[1], X.shape[1])))
    assert linear_cka(X, 3.0 * X @ Q) == pytest.approx(1.0, abs=1e-5)


def test_linear_cka_is_symmetric(paired_data: tuple[np.ndarray, np.ndarray]) -> None:
    """CKA gives the same value regardless of argument order."""
    X, Y = paired_data
    assert linear_cka(X, Y) == pytest.approx(linear_cka(Y, X), rel=1e-6)


def test_gram_cka_matches_feature_space_cka(paired_data: tuple[np.ndarray, np.ndarray]) -> None:
    """The Gram-matrix (PyTorch) CKA equals the feature-space (NumPy) CKA."""
    X, Y = paired_data
    true_cka, *_ = evaluate_alignment_exact(X, Y, n_permutations=5, device=CPU)
    assert true_cka == pytest.approx(linear_cka(X.astype(np.float64), Y.astype(np.float64)), rel=1e-6)


def test_cka_permutation_test_separates_signal_from_noise(paired_data: tuple[np.ndarray, np.ndarray]) -> None:
    """The CKA permutation test flags shared signal and does not flag unrelated noise."""
    X, Y = paired_data
    _, _, adj, p = evaluate_alignment_exact(X, Y, n_permutations=50, device=CPU)
    assert adj > 0.1 and p == pytest.approx(1 / 51)

    noise = np.random.default_rng(2).normal(size=Y.shape)
    _, _, adj_noise, p_noise = evaluate_alignment_exact(X, noise, n_permutations=50, device=CPU)
    # Unrelated data: negligible adjusted signal and not at the permutation floor.
    assert abs(adj_noise) < 0.05 and p_noise > 1 / 51


def test_numpy_and_torch_cka_agree(paired_data: tuple[np.ndarray, np.ndarray]) -> None:
    """The NumPy and PyTorch CKA implementations report the same true CKA."""
    X, Y = paired_data
    np.random.seed(0)
    true_np, *_ = evaluate_alignment(X.astype(np.float64), Y.astype(np.float64), n_permutations=3)
    true_t, *_ = evaluate_alignment_exact(X, Y, n_permutations=3, device=CPU)
    assert true_np == pytest.approx(true_t, rel=1e-6)


def test_svd_flip_makes_largest_entry_positive_and_is_sign_invariant() -> None:
    """Sign flipping makes each column's largest entry positive, whatever the input sign."""
    u = torch.tensor(np.random.default_rng(3).normal(size=(20, 4)))
    flipped = svd_flip_torch(u)
    idx = torch.argmax(torch.abs(flipped), dim=0)
    assert torch.all(flipped[idx, torch.arange(4)] > 0)
    assert torch.allclose(svd_flip_torch(-u), flipped)


def test_svd_flip_applies_same_signs_to_v() -> None:
    """The rows of V are flipped with the same signs as the columns of U."""
    rng = np.random.default_rng(4)
    u = torch.tensor(rng.normal(size=(6, 3)))
    v = torch.tensor(rng.normal(size=(3, 5)))
    u_new, v_new = svd_flip_torch(u, v.clone())
    signs = (u_new / u)[0]
    assert torch.allclose(v_new, v * signs.unsqueeze(1))


def test_subspace_basis_is_orthonormal_and_spans_top_pcs(paired_data: tuple[np.ndarray, np.ndarray]) -> None:
    """The Gram-matrix basis is orthonormal and spans the same space as the top principal components."""
    X, _ = paired_data
    Xc = X.astype(np.float64) - X.mean(axis=0)
    Q = get_exact_subspace_basis(torch.tensor(Xc), k=8).numpy()
    np.testing.assert_allclose(Q.T @ Q, np.eye(8), atol=1e-8)
    U, _, _ = np.linalg.svd(Xc, full_matrices=False)
    # Same subspace: projection matrices agree.
    np.testing.assert_allclose(Q @ Q.T, U[:, :8] @ U[:, :8].T, atol=1e-6)


def test_variance_threshold_basis_reaches_threshold(paired_data: tuple[np.ndarray, np.ndarray]) -> None:
    """The variance-threshold basis keeps enough components to reach the requested variance."""
    X, _ = paired_data
    Xc = torch.tensor(X.astype(np.float64) - X.mean(axis=0))
    _, k, var = get_exact_subspace_basis_var_thresh(Xc, variance_threshold=0.9)
    _, _, var_prev = get_exact_subspace_basis_var_thresh(Xc, variance_threshold=0.0)
    assert var >= 0.9
    assert k >= 5 and var_prev <= var


def test_svcca_equals_classical_cca_on_principal_subspaces(paired_data: tuple[np.ndarray, np.ndarray]) -> None:
    """SVCCA canonical correlations equal classical CCA computed on the top principal components."""
    X, Y = paired_data
    k = 10
    corrs = get_svcca_decay_exact(X, Y, k_dims=k, device=CPU)
    Ux = np.linalg.svd(X.astype(np.float64) - X.mean(0), full_matrices=False)[0][:, :k]
    Uy = np.linalg.svd(Y.astype(np.float64) - Y.mean(0), full_matrices=False)[0][:, :k]
    reference = np.linalg.svd(Ux.T @ Uy, compute_uv=False)
    np.testing.assert_allclose(corrs, np.sort(reference)[::-1], atol=1e-6)
    # The 5 shared latent dimensions give 5 near-perfect canonical correlations.
    assert np.all(corrs[:5] > 0.9) and corrs[6] < 0.9


def test_svcca_null_detects_shared_axes(paired_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Both SVCCA-with-null variants find at least the 5 planted shared axes."""
    X, Y = paired_data
    true_corrs, null_mean, null_std, sig_axes = get_svcca_decay_and_null_exact(X, Y, k_dims=10, n_perms=30, device=CPU)
    assert true_corrs.shape == null_mean.shape == null_std.shape == (10,)
    assert sig_axes >= 5

    out = get_svcca_with_model_null_exact(X, Y, var_thresh=0.9, n_perms=30, device=CPU)
    true_corrs, null_mean, null_std, sig_axes, k_x, k_y, var_x, var_y = out
    assert len(true_corrs) == min(k_x, k_y)
    assert var_x >= 0.9 and var_y >= 0.9 and sig_axes >= 5


def test_compress_passthrough_and_dtype(paired_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Matrices already small enough are returned unchanged, in the requested dtype."""
    X, _ = paired_data
    out = compress_to_exact_dimension(X, target_dim=384, device=CPU)
    assert out.dtype == np.float32 and np.array_equal(out, X)
    assert compress_to_exact_dimension(X, 384, dtype=np.float64, device=CPU).dtype == np.float64


def test_compress_preserves_geometry_when_all_components_kept() -> None:
    """PCA compression keeps pairwise distances when no variance is discarded."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 300))
    Z = compress_to_exact_dimension(X, target_dim=100, dtype=np.float64, device=CPU)
    # With fewer compounds than target_dim, only n_compounds components exist.
    assert Z.shape == (40, 40)
    Xc = X - X.mean(axis=0)
    # PCA is a rotation of the centred data: pairwise distances are preserved.
    d_orig = np.linalg.norm(Xc[:, None] - Xc[None], axis=-1)
    d_new = np.linalg.norm(Z[:, None] - Z[None], axis=-1)
    np.testing.assert_allclose(d_new, d_orig, atol=1e-2)


@pytest.mark.parametrize("mode, expected_dims", [("none", 60), ("fixed", 10)])
def test_reduce_dimensions_none_and_fixed(
    paired_data: tuple[np.ndarray, np.ndarray], mode: str, expected_dims: int
) -> None:
    """'none' keeps every feature, 'fixed' keeps the requested number of components."""
    X, _ = paired_data
    out = reduce_dimensions(X, mode=mode, fixed_dims=10, device=CPU)
    assert out.shape == (X.shape[0], expected_dims) and out.dtype == np.float32


def test_reduce_dimensions_variance_keeps_the_signal(paired_data: tuple[np.ndarray, np.ndarray]) -> None:
    """The variance mode keeps few components for low-rank data and preserves distances between compounds."""
    X, _ = paired_data
    out = reduce_dimensions(X, mode="variance", var_threshold=0.99, device=CPU)
    assert 5 <= out.shape[1] < X.shape[1]
    centered = X - X.mean(axis=0)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), np.linalg.norm(centered, axis=1), rtol=0.1)


def test_reduce_dimensions_rejects_unknown_mode(paired_data: tuple[np.ndarray, np.ndarray]) -> None:
    """An unknown mode raises a clear error."""
    with pytest.raises(ValueError, match="mode must be"):
        reduce_dimensions(paired_data[0], mode="bogus", device=CPU)


def test_svcca_with_cutoff_modes(paired_data: tuple[np.ndarray, np.ndarray]) -> None:
    """The noise floor lies above the null; a stricter floor never finds more significant axes."""
    X, Y = paired_data
    true, null_mean, null_std, floor_3sd, sig_3sd = get_svcca_decay_with_cutoff_exact(
        X, Y, k_dims=20, n_perms=30, sigma_val=3.0, device=CPU
    )
    assert np.all(floor_3sd >= null_mean) and len(floor_3sd) == len(true)
    assert sig_3sd >= 4
    _, _, _, _, sig_1sd = get_svcca_decay_with_cutoff_exact(X, Y, k_dims=20, n_perms=30, sigma_val=1.0, device=CPU)
    assert sig_1sd >= sig_3sd
    _, _, _, floor_pct, sig_pct = get_svcca_decay_with_cutoff_exact(
        X, Y, k_dims=20, n_perms=30, thresh_mode="percentile", pct_val=95.0, device=CPU
    )
    assert np.all(floor_pct >= null_mean - 1e-12) and sig_pct >= 4


def test_svcca_with_cutoff_matches_default_two_sd(paired_data: tuple[np.ndarray, np.ndarray]) -> None:
    """With a 2-SD floor the result agrees with the existing fixed-rule function."""
    X, Y = paired_data
    true, null_mean, null_std, sig = get_svcca_decay_and_null_exact(X, Y, k_dims=20, n_perms=30, device=CPU)
    true2, null_mean2, _, floor, sig2 = get_svcca_decay_with_cutoff_exact(
        X, Y, k_dims=20, n_perms=30, sigma_val=2.0, device=CPU
    )
    np.testing.assert_allclose(true, true2)
    np.testing.assert_allclose(null_mean, null_mean2)
    assert sig == sig2


def test_svcca_with_cutoff_rejects_unknown_mode(paired_data: tuple[np.ndarray, np.ndarray]) -> None:
    """An unknown threshold mode raises a clear error."""
    with pytest.raises(ValueError, match="thresh_mode"):
        get_svcca_decay_with_cutoff_exact(*paired_data, thresh_mode="bogus", device=CPU)
