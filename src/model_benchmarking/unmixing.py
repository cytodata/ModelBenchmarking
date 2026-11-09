"""Separate the entangled canonical axes into more interpretable programs.

Canonical axes from SVCCA are orthogonal combinations of whatever varies jointly in the two
spaces, so one axis can mix several biological programs. Three optional "unmixing" steps
re-express the same subspace:

* **Varimax** rotates the axes (orthogonally) so that each compound loads strongly on few axes.
* **FastICA** finds statistically independent sources (maximizing non-Gaussianity).
* **Hybrid** applies FastICA, then Varimax to the independent sources.

None of these changes which subspace is described, only how it is split into axes; the
canonical correlation of an original axis therefore no longer belongs to an unmixed axis.
"""

import numpy as np
from scipy.stats import skew
from sklearn.decomposition import FastICA

UNMIXING_METHODS = ("none", "varimax", "fastica", "hybrid")
AXIS_PREFIXES = {"none": "Bio_Axis", "varimax": "Rotated_Axis", "fastica": "ICA_Axis", "hybrid": "Hybrid_Axis"}


def varimax_rotation(
    scores: np.ndarray, gamma: float = 1.0, max_iter: int = 50, tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal Varimax rotation, which maximizes the sparsity of the loadings.

    Parameters
    ----------
    scores : numpy.ndarray, shape (n_compounds, n_axes)
    gamma : float, default 1.0
        1 gives the classical Varimax criterion.
    max_iter : int, default 50
    tol : float, default 1e-6
        Stop when the criterion improves by less than this relative amount.

    Returns
    -------
    rotated : numpy.ndarray, shape (n_compounds, n_axes)
    rotation : numpy.ndarray, shape (n_axes, n_axes)
        Orthogonal matrix with ``rotated = scores @ rotation``.
    """
    p, k = scores.shape
    rotation = np.eye(k)
    criterion = 0.0
    for _ in range(max_iter):
        previous = criterion
        loadings = scores @ rotation
        u, s, vh = np.linalg.svd(
            scores.T @ (loadings**3 - (gamma / p) * loadings @ np.diag(np.sum(loadings**2, axis=0)))
        )
        rotation = u @ vh
        criterion = float(np.sum(s))
        if previous != 0 and criterion / previous < 1 + tol:
            break
    return scores @ rotation, rotation


def _orient_and_sort(components: np.ndarray) -> np.ndarray:
    """Make each component's skew positive and order components by decreasing variance."""
    signs = np.where(skew(components, axis=0) < 0, -1.0, 1.0)
    components = components * signs
    return components[:, np.argsort(np.var(components, axis=0))[::-1]]


def apply_fastica_unmixing(
    scores: np.ndarray, random_state: int = 42, max_iter: int = 1000, tol: float = 1e-4
) -> np.ndarray:
    """Decompose axes into statistically independent components (FastICA, log-cosh contrast).

    Components are oriented so that their skewness is positive (their long tail points to
    high scores) and sorted by decreasing variance.

    Returns
    -------
    numpy.ndarray, shape (n_compounds, n_axes)
    """
    ica = FastICA(
        n_components=scores.shape[1],
        algorithm="parallel",
        whiten="unit-variance",
        fun="logcosh",
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    return _orient_and_sort(ica.fit_transform(scores))


def apply_hybrid_unmixing(
    scores: np.ndarray, random_state: int = 42, max_iter: int = 1000, tol: float = 1e-4
) -> np.ndarray:
    """FastICA followed by Varimax, with the same orientation and ordering as FastICA."""
    independent = apply_fastica_unmixing(scores, random_state=random_state, max_iter=max_iter, tol=tol)
    rotated, _ = varimax_rotation(independent)
    return _orient_and_sort(rotated)


def unmix_scores(scores: np.ndarray, method: str = "none", random_state: int = 42) -> tuple[np.ndarray, str]:
    """Apply the chosen unmixing method and return the scores with a column-name prefix.

    Parameters
    ----------
    scores : numpy.ndarray, shape (n_compounds, n_axes)
        Compound scores on the canonical axes.
    method : {"none", "varimax", "fastica", "hybrid"}, default "none"
    random_state : int, default 42
        Seed for FastICA.

    Returns
    -------
    unmixed : numpy.ndarray, shape (n_compounds, n_axes)
    prefix : str
        ``"Bio_Axis"``, ``"Rotated_Axis"``, ``"ICA_Axis"`` or ``"Hybrid_Axis"``, for naming the columns.

    Raises
    ------
    ValueError
        For an unknown ``method``.
    """
    if method not in UNMIXING_METHODS:
        raise ValueError(f"method must be one of {UNMIXING_METHODS}, received: {method!r}")
    if method == "varimax":
        unmixed, _ = varimax_rotation(scores)
    elif method == "fastica":
        unmixed = apply_fastica_unmixing(scores, random_state=random_state)
    elif method == "hybrid":
        unmixed = apply_hybrid_unmixing(scores, random_state=random_state)
    else:
        unmixed = scores
    return unmixed, AXIS_PREFIXES[method]
