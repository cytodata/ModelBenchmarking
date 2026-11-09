"""Tests for model_benchmarking.plotting and model_benchmarking.utils."""

import io
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
import torch

from model_benchmarking import plotting
from model_benchmarking.utils import DownloadProgressBar, get_device, log_debug, release_memory

K_LIST = [1, 5, 10]


@pytest.fixture
def knn_results() -> pd.DataFrame:
    """A small pairwise kNN result table comparing three datasets at three k values."""
    rows = []
    for d1, d2 in [("Minimol", "DINO_Raw"), ("Minimol", "CC_Sign3"), ("CC_Sign3", "DINO_Raw")]:
        for k in K_LIST:
            rows.append({"Dataset_1": d1, "Dataset_2": d2, "k": k, "Overlap_Pct": 1.0 + k, "Fold_Enrichment": 2.0 * k})
    return pd.DataFrame(rows)


def test_plot_heatmap_orders_datasets(knn_results: pd.DataFrame) -> None:
    """The heatmap is drawn with datasets in the structure -> bioactivity -> morphology order."""
    plotting.plot_heatmap(knn_results, "Fold_Enrichment", "title", ["Minimol", "CC_Sign3", "DINO_Raw"])
    ax = plt.gcf().axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Minimol", "CC_Sign3", "DINO_Raw"]


def test_plot_anchor_sweep_draws_one_line_per_comparison(knn_results: pd.DataFrame) -> None:
    """The single-panel k-sweep draws one line for each dataset compared with the anchor."""
    plotting.plot_anchor_sweep(knn_results, "Minimol", K_LIST, sns.color_palette("husl", 2))
    assert len(plt.gca().get_lines()) >= 2
    assert plt.gca().get_xscale() == "log"


@pytest.mark.parametrize("plot", [plotting.plot_dual_metrics, plotting.plot_anchor_sweep_dual])
def test_dual_panel_plots(knn_results: pd.DataFrame, plot: Callable[..., None]) -> None:
    """Both two-panel plots show overlap on the left and fold enrichment on the right."""
    plot(knn_results, "CC_Sign3", K_LIST, sns.color_palette("Set2", 2))
    left, right = plt.gcf().axes[:2]
    assert left.get_ylabel() == "Unique Edge Overlap (%)"
    assert "Fold Enrichment" in right.get_ylabel()


def test_anchor_sweep_warns_when_anchor_missing(knn_results: pd.DataFrame, capsys: pytest.CaptureFixture) -> None:
    """Plotting an anchor that has no comparisons prints a warning instead of drawing."""
    plotting.plot_anchor_sweep(knn_results, "Unknown", K_LIST, "husl")
    assert "No successful alignments" in capsys.readouterr().out


def test_plot_svcca_decay_finds_shared_axes() -> None:
    """The SVCCA pilot plot detects the planted shared axes and returns one correlation per axis."""
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(200, 3))
    X = latent @ rng.normal(size=(3, 20)) + 0.3 * rng.normal(size=(200, 20))
    Y = latent @ rng.normal(size=(3, 15)) + 0.3 * rng.normal(size=(200, 15))
    np.random.seed(0)
    true_corrs, null_mean, sig = plotting.plot_svcca_decay(X, Y, n_components=10, n_permutations=5)
    assert true_corrs.shape == null_mean.shape == (10,)
    assert sig >= 3


def test_get_device_returns_usable_device() -> None:
    """The selected device can hold a tensor."""
    device = get_device()
    assert isinstance(device, torch.device)
    assert torch.zeros(1, device=device).device.type == device.type


def test_log_debug_prints_message(capsys: pytest.CaptureFixture) -> None:
    """log_debug prints the message together with RAM usage."""
    log_debug("hello", t_start=0.0)
    out = capsys.readouterr().out
    assert "hello" in out and "RAM" in out


def test_download_progress_bar_tracks_bytes() -> None:
    """The urlretrieve hook converts block counts into bytes downloaded."""
    with DownloadProgressBar(file=io.StringIO()) as bar:
        bar.update_to(b=3, bsize=100, tsize=1000)
        assert bar.n == 300 and bar.total == 1000


@pytest.fixture
def svcca_results() -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Leaderboard, decay curves and null curves for three models."""
    names = ["OpenPhenom 384 (Normed)", "DINO (Raw)", "Imagenet (normalized)"]
    decay = {n: np.linspace(0.9 - 0.1 * i, 0.1, 50) for i, n in enumerate(names)}
    nulls = {n: (np.linspace(0.4, 0.05, 50), np.full(50, 0.02)) for n in names}
    leaderboard = pd.DataFrame(
        {"Model / Processing": names, "SVCCA Adj. AUC": [0.3, 0.2, 0.1], "Significant Axes": [10, 8, 5]}
    )
    return leaderboard, decay, nulls


def test_svcca_overview_plots(
    svcca_results: tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]],
) -> None:
    """Both overview figures draw one curve per model plus the null baseline."""
    _, decay, nulls = svcca_results
    plotting.plot_svcca_decay_overview(decay, nulls, "DINO (Raw)", max_axes=50, title="t")
    assert len(plt.gcf().axes[0].get_lines()) == len(decay) + 1
    plt.close("all")
    plotting.plot_svcca_decay_overview_v2(decay, *nulls["DINO (Raw)"])
    assert len(plt.gcf().axes[0].get_lines()) == len(decay) + 1


def test_svcca_grid_one_panel_per_model(
    svcca_results: tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]],
) -> None:
    """The decay grid has one titled panel per model, in leaderboard order."""
    leaderboard, decay, nulls = svcca_results
    plotting.plot_svcca_decay_grid(leaderboard, decay, nulls)
    titles = [ax.get_title() for ax in plt.gcf().axes]
    assert len(titles) == 3 and titles[0].startswith("Rank 1: OpenPhenom 384 (Normed)")


def test_release_memory_runs() -> None:
    """Releasing memory never raises, even where glibc's malloc_trim is unavailable."""
    release_memory()


def test_svcca_floor_grid_one_panel_per_model(
    svcca_results: tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]],
) -> None:
    """The floor grid has one panel per model with the benchmark's noise floor drawn from the null."""
    leaderboard, decay, nulls = svcca_results
    leaderboard = leaderboard.assign(**{"Compounds (N)": [100, 200, 300], "Evaluated Dims": [10, 20, 30]})
    cutoffs = {name: nulls[name][0] + 3 * nulls[name][1] for name in nulls}
    plotting.plot_svcca_floor_grid(leaderboard, decay, nulls, cutoffs, "Noise Floor (+3σ)", "settings", ncols=2)
    panels = [ax for ax in plt.gcf().axes if ax.get_title()]
    assert len(panels) == 3 and panels[0].get_title().startswith("OpenPhenom 384 (Normed)\nN=100 | D=10")
    assert "Noise Floor (+3σ)" in [t.get_text() for t in panels[0].get_legend().get_texts()]
