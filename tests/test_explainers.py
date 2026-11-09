"""Tests for model_benchmarking.explainers (gene-module and reference-compound scans of the axes)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from model_benchmarking import explainers


def test_reference_keys_are_well_formed() -> None:
    """Every reference InChIKey has the standard 27-character shape and a unique first block."""
    keys = list(explainers.TUBULIN_DRUGS)
    assert all(len(k) == 27 and k[14] == "-" and k[25] == "-" for k in keys)
    assert len({k[:14] for k in keys}) == len(keys)
    assert all({"name", "class"} <= set(v) for v in explainers.TUBULIN_DRUGS.values())


def test_resolve_gene_columns() -> None:
    """Symbols are matched through Ensembl IDs (version suffix ignored) or directly as column names."""
    cols = ["ENSG1.5", "ENSG2", "TUBB"]
    matched = explainers.resolve_gene_columns(
        ["plk1", "Aurka", "TUBB", "MISSING"], cols, {"ENSG1": "PLK1", "ENSG2": "AURKA"}
    )
    assert matched == {"PLK1": "ENSG1.5", "AURKA": "ENSG2", "TUBB": "TUBB"}


def test_module_scores_average_z_scores() -> None:
    """Module scores are the mean z-score; constant genes and NaNs do not break them."""
    expr = np.array([[1.0, 5.0, np.nan], [2.0, 5.0, 1.0], [3.0, 5.0, 1.0]])
    scores = explainers.module_scores(expr)
    assert scores.shape == (3,) and np.all(np.isfinite(scores)) and np.argmax(scores) == 2
    assert scores.mean() == pytest.approx(0.0, abs=1e-12)


@pytest.fixture
def scan_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Two axes: one correlated (negatively) with a module score, one unrelated."""
    rng = np.random.default_rng(0)
    module = rng.normal(size=200)
    axes = pd.DataFrame({"Axis_1": rng.normal(size=200), "Axis_2": -module + 0.2 * rng.normal(size=200)})
    return axes, module


def test_scan_axes_for_module(scan_data: tuple[pd.DataFrame, np.ndarray]) -> None:
    """The anti-correlated axis ranks first with a negative r and a tiny p-value."""
    axes, module = scan_data
    scan = explainers.scan_axes_for_module(axes, module, canonical_r=[0.3, 0.2])
    assert scan.loc[0, "Axis"] == "Axis_2" and scan.loc[0, "Module_Corr_Pearson"] < -0.9
    assert scan.loc[0, "P_Value"] < 1e-10 and scan.loc[0, "Canonical_r"] == 0.2


def test_scan_axes_for_module_handles_constant_axis() -> None:
    """A constant axis gets zero correlation instead of an error."""
    scan = explainers.scan_axes_for_module(pd.DataFrame({"A": np.ones(10)}), np.arange(10.0))
    assert scan.loc[0, "Module_Corr_Pearson"] == 0.0 and scan.loc[0, "P_Value"] == 1.0


def test_match_reference_compounds() -> None:
    """Exact keys and keys that only share the first block are both found."""
    paclitaxel = next(k for k, v in explainers.TUBULIN_DRUGS.items() if v["name"] == "Paclitaxel")
    variant = paclitaxel[:14] + "-XXXXXXXXXX-N"
    flags, details = explainers.match_reference_compounds(
        [paclitaxel, variant, "AAAAAAAAAAAAAA-BBBBBBBBBB-C"], explainers.TUBULIN_DRUGS
    )
    assert flags.tolist() == [True, True, False]
    assert details[0] == ("Paclitaxel", "Taxane (Stabilizer)", "Exact 27-char")
    assert details[1][2] == "Block-1 Rescue" and details[2] is None


@pytest.fixture
def reference_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Axis 1 separates a reference group (higher scores), axis 2 does so in the opposite direction."""
    rng = np.random.default_rng(1)
    n = 200
    is_ref = np.zeros(n, dtype=bool)
    is_ref[:15] = True
    a1 = rng.normal(size=n) + 3 * is_ref
    a2 = rng.normal(size=n) - 3 * is_ref
    a3 = rng.normal(size=n)
    return pd.DataFrame({"A1": a1, "A2": a2, "A3": a3}), is_ref


def test_scan_axes_for_reference_set(reference_data: tuple[pd.DataFrame, np.ndarray]) -> None:
    """Both separating axes reach high AUC with the right polarity; the noise axis stays near chance."""
    axes, is_ref = reference_data
    scan = explainers.scan_axes_for_reference_set(axes, is_ref).set_index("Axis")
    assert scan.loc["A1", "Polarity"] == "Positive (+)" and scan.loc["A2", "Polarity"] == "Negative (-)"
    assert scan.loc["A1", "ROC_AUC"] > 0.9 and scan.loc["A2", "ROC_AUC"] > 0.9
    assert scan.loc["A3", "ROC_AUC"] < 0.7 and scan.loc["A3", "P_Value"] > 0.01
    assert scan.loc["A1", "Cohen_d"] > 0 > scan.loc["A2", "Cohen_d"]


def test_reference_module_separation() -> None:
    """Reference compounds with a higher module score are significantly separated; too few compounds give None."""
    module = np.concatenate([np.full(10, 3.0) + np.arange(10) * 0.01, np.zeros(40) + np.arange(40) * 0.001])
    is_ref = np.arange(50) < 10
    ref_mean, bg_mean, p = explainers.reference_module_separation(module, is_ref)
    assert ref_mean > bg_mean and p < 1e-6
    assert explainers.reference_module_separation(module, np.arange(50) == 0) is None


def test_print_module_drivers(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Drivers are listed from the highest oriented axis score, annotated."""
    monkeypatch.setattr(explainers, "get_rich_compound_annotation", lambda key: (f"name-{key}", "MOA"))
    table = pd.DataFrame({"InChIKey": ["K1", "K2", "K3"], "Axis": [-3.0, 1.0, -1.0]})
    explainers.print_module_drivers(table, "Axis", -1.0, np.array([0.5, -0.1, 0.2]), "InChIKey", top_n=2)
    lines = capsys.readouterr().out.strip().splitlines()
    assert "name-K1" in lines[0] and "Axis: +3.000" in lines[0] and "name-K3" in lines[1]


def test_plots(scan_data: tuple[pd.DataFrame, np.ndarray], reference_data: tuple[pd.DataFrame, np.ndarray]) -> None:
    """Both scan figures have two panels."""
    axes, module = scan_data
    explainers.plot_module_scan(axes, module, explainers.scan_axes_for_module(axes, module))
    assert len(plt.gcf().axes) == 2
    ref_axes, is_ref = reference_data
    explainers.plot_reference_scan(ref_axes, is_ref, explainers.scan_axes_for_reference_set(ref_axes, is_ref))
    assert len(plt.gcf().axes) == 2
