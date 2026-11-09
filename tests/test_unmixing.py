"""Tests for model_benchmarking.unmixing (Varimax, FastICA and hybrid axis unmixing)."""

import numpy as np
import pytest

from model_benchmarking import unmixing


@pytest.fixture
def mixed_sources() -> np.ndarray:
    """Three independent, non-Gaussian sources linearly mixed into three axes."""
    rng = np.random.default_rng(0)
    sources = np.column_stack([rng.laplace(size=400), rng.exponential(size=400), rng.uniform(-1, 1, size=400)])
    return sources @ rng.normal(size=(3, 3))


def test_varimax_rotation_is_orthogonal_and_preserves_the_subspace(mixed_sources: np.ndarray) -> None:
    """Rotating gives back the input through the returned orthogonal matrix, and keeps row norms."""
    rotated, rotation = unmixing.varimax_rotation(mixed_sources)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
    np.testing.assert_allclose(rotated, mixed_sources @ rotation)
    np.testing.assert_allclose(np.linalg.norm(rotated, axis=1), np.linalg.norm(mixed_sources, axis=1))


def test_varimax_recovers_sparse_axes() -> None:
    """Varimax undoes a rotation of two sparse axes, up to sign and order."""
    rng = np.random.default_rng(1)
    sparse = np.zeros((300, 2))
    sparse[:20, 0] = rng.normal(5, 1, 20)
    sparse[20:40, 1] = rng.normal(5, 1, 20)
    theta = 0.4
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated, _ = unmixing.varimax_rotation(sparse @ rot)
    cross = np.abs(np.corrcoef(rotated.T, sparse.T)[:2, 2:])
    assert np.sort(cross.max(axis=1)).min() > 0.99 and cross.argmax(axis=1).tolist() in ([0, 1], [1, 0])


def test_fastica_recovers_independent_sources(mixed_sources: np.ndarray) -> None:
    """FastICA components are uncorrelated, oriented by positive skew and sorted by variance."""
    out = unmixing.apply_fastica_unmixing(mixed_sources)
    assert out.shape == mixed_sources.shape
    off_diag = np.corrcoef(out.T) - np.eye(3)
    assert np.abs(off_diag).max() < 0.2
    variances = out.var(axis=0)
    assert np.all(np.diff(variances) <= 1e-9)


def test_fastica_is_deterministic(mixed_sources: np.ndarray) -> None:
    """The same seed gives the same components."""
    np.testing.assert_allclose(
        unmixing.apply_fastica_unmixing(mixed_sources), unmixing.apply_fastica_unmixing(mixed_sources)
    )


def test_hybrid_shape_and_ordering(mixed_sources: np.ndarray) -> None:
    """The hybrid method returns the same number of axes, ordered by variance."""
    out = unmixing.apply_hybrid_unmixing(mixed_sources)
    assert out.shape == mixed_sources.shape
    assert np.all(np.diff(out.var(axis=0)) <= 1e-9)


@pytest.mark.parametrize(
    "method, prefix",
    [("none", "Bio_Axis"), ("varimax", "Rotated_Axis"), ("fastica", "ICA_Axis"), ("hybrid", "Hybrid_Axis")],
)
def test_unmix_scores_dispatch(mixed_sources: np.ndarray, method: str, prefix: str) -> None:
    """Every method returns scores of the same shape and the matching column-name prefix."""
    out, got_prefix = unmixing.unmix_scores(mixed_sources, method)
    assert out.shape == mixed_sources.shape and got_prefix == prefix
    if method == "none":
        np.testing.assert_array_equal(out, mixed_sources)


def test_unmix_scores_rejects_unknown_method(mixed_sources: np.ndarray) -> None:
    """An unknown method raises a clear error."""
    with pytest.raises(ValueError, match="method must be one of"):
        unmixing.unmix_scores(mixed_sources, "pca")
